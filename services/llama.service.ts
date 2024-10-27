import 'dotenv/config';
import ollama from 'ollama';
import { VoyageAIClient } from 'voyageai';
import { createMessage, getAllMessages } from '../models/message.model';
import { searchIndex } from './embedder.service';

const voyageClient = new VoyageAIClient({ apiKey: process.env.VOYAGE_API_KEY });

export async function sendMessage(threadId: string, content: { text: string; url?: string }) {
    console.log('sendMessage', threadId, JSON.stringify(content));
    //Set default response
    let response = { content: 'Sorry, I am not able to understand your question. Please try again.' };

    // 1. Load history of messages from the database
    const dbMessages = await getAllMessages(threadId);

    // 2. Add user message to the database
    await createMessage({
        thread_id: threadId,
        role: 'user',
        content: content.text,
        url: content.url || null,
    });

    // 3. Prepare messages for OpenAI
    const messages: any[] = dbMessages.map(message => ({
        role: message.role,
        content: message.content,
    }));
    messages.push({
        role: 'user',
        content: content.text,
    });

    // 4. Create a LLAMA message
    const msg = await createLlamaMessage(messages);
    if (msg.tool_calls) {
        messages.push(msg);
        console.log('tool_calls', msg.tool_calls);
        const toolCall = msg.tool_calls[0];
        const embeddingResult = await embed(toolCall.function.arguments.question);
        const vector = embeddingResult[0].embedding;
        const matches = (await searchIndex(vector)).matches;
        if (matches.length) {
            // console.log(matches);
            console.log('Found matches:', matches.length);
            const context = matches.map(match => match.metadata.content).join('\n\n');
            // console.log('Context:', context);
            messages.push({
                role: 'tool',
                content: context,
            });
        } else {
            messages.push({
                role: 'tool',
                content: '',
            });
        }
        const newMsg = await createLlamaMessage(messages);
        console.log(newMsg);
        response.content = newMsg.content;
    } else {
        response.content = msg.content;
    }

    // 5. Add assistant message to the database
    await createMessage({
        thread_id: threadId,
        role: 'assistant',
        content: response.content,
    });

    return response;
}

async function createLlamaMessage(messages: any[]) {
    const instructions =
        "You are a customer support assistant for TechEase Solutions, a company that provides comprehensive IT services to businesses. Your role is to assist our clients with their technical issues, answer questions about our services, and provide guidance on using our products effectively. Always respond in a friendly, professional manner, and ensure your explanations are clear and concise. If you're unable to resolve an issue immediately, reassure the customer that you will escalate the problem and follow up promptly. Your goal is to provide exceptional support and ensure customer satisfaction. Do not mention your tools";

    const msg = await ollama.chat({
        model: 'llama3.1',
        messages: [{ role: 'system', content: instructions }, ...messages],
        tools: [
            {
                type: 'function',
                function: {
                    name: 'getContext',
                    description: "Always retrieve relevant context to answer a user's IT/software related question",
                    parameters: {
                        type: 'object',
                        properties: {
                            question: {
                                type: 'string',
                                description: "The user's question is about IT or software and requires additional context.",
                            },
                        },
                        required: ['question'],
                    },
                },
            },
        ],
    });

    return msg.message;
}

export async function embed(text: string) {
    return (
        await voyageClient.embed({
            input: text,
            model: 'voyage-3-lite',
        })
    ).data;
}
