import { readFileSync } from 'fs';
import { join } from 'path';
import { v4 as uuidv4 } from 'uuid';
import { embed } from '../services/anthropic.service';
import { upsertToIndex } from '../services/embedder.service';
import { splitTextIntoChunks } from '../services/utilitiess.service';

async function main() {
    // 1. Read the knowledge base from the file (./../kdb.txt)
    const contents = readFileSync(join(__dirname, './../kdb.txt'), 'utf8');
    console.log(contents.length);

    // 2. Split the text into paragraphs
    // const paragraphs = contents.split('\n\n');

    // 2. Split the text into chunks
    const chunks = splitTextIntoChunks(contents);

    // 3. Create embeddings for each chunk
    let i = 0;
    const len = chunks.length;
    const data = [];
    for (const chunk of chunks) {
        i++;
        console.log(`Processing chunk ${i} of ${len}`);

        const result = await embed(chunk);
        console.log(result);
        data.push({
            id: uuidv4(),
            values: result[0].embedding,
            metadata: {
                content: chunk,
            },
        });
    }

    console.log(data);
    await upsertToIndex(data);
}

main().catch(console.error);
