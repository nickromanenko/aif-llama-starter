import { embed } from './services/anthropic.service';

async function main() {
    const result = await embed('How do I install a package in Python?');
    console.log(result);
}

main().catch(console.error);
