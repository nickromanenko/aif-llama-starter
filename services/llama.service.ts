import 'dotenv/config';
import { VoyageAIClient } from 'voyageai';
import { createMessage, getAllMessages } from '../models/message.model';

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

    // 5. Add assistant message to the database
    await createMessage({
        thread_id: threadId,
        role: 'assistant',
        content: response.content,
    });

    return response;
}

async function createLlamaMessage(messages: any[]) {}

export async function embed(text: string) {
    return (
        await voyageClient.embed({
            input: text,
            model: 'voyage-3-lite',
        })
    ).data;
}
