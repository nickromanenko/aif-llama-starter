import { Pinecone } from '@pinecone-database/pinecone';
import 'dotenv/config';

const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
});
const index = pc.index('techease2');

export async function upsertToIndex(data: any[]) {
    return await index.upsert(data);
}

export async function searchIndex(vector: number[]) {
    // console.log('Searching index with vector:', vector);
    return await index.query({ vector, topK: 3, includeMetadata: true });
}
