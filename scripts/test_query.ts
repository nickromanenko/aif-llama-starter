import { embed } from '../services/anthropic.service';
import { searchIndex } from '../services/embedder.service';

async function main() {
    const query = 'How to resolve Error Code 6006?';
    const embeddingResult = await embed(query);

    const vector = embeddingResult[0].embedding;

    const matches = (await searchIndex(vector)).matches.filter(match => match.score > 0.5);

    for (const match of matches) {
        console.log(match.metadata.content);
    }
}

main().catch(console.error);
