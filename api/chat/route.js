import { NextResponse } from "next/server";
import { Pinecone } from '@pinecone-database/pinecone';
import OpenAI from 'openai';

const systemPrompt = `
You are an AI assistant designed to help students find professors based on their specific queries. Your primary function is to provide information about the top 3 most relevant professors for each student's request.

Your responses should follow this format:

1. Brief acknowledgment of the student's query
2. List of the top 3 professors, each including:
   - Name
   - Department
   - Overall rating (out of 5)
   - Brief summary of strengths and weaknesses
   - A short, representative student quote

3. A concise conclusion or recommendation based on the results

Use a Retrieval-Augmented Generation (RAG) system to access and provide the most up-to-date and relevant information from the professor database. This ensures that your responses are based on current data and student feedback.

Remember to maintain a neutral and informative tone. Your goal is to provide accurate information to help students make informed decisions, not to promote or discourage selecting any particular professor.

If a query is too vague or broad, ask for clarification to ensure you provide the most relevant results. If there aren't enough professors matching the criteria, inform the student and provide the best available options.

Prioritize factors such as teaching quality, course difficulty, grading fairness, and overall student satisfaction in your recommendations. Be prepared to explain your reasoning if asked.

Maintain student and professor privacy by not sharing personal information beyond what's publicly available in standard professor reviews.

Always encourage students to do further research and consider their own learning style and academic goals when making decisions about course selection.
`;

export async function POST(req) {
    const data = await req.json();
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API,
    });
    const index = pc.index('rate-my-professor-rag').namespace('ns1');
    const openai = new OpenAI(process.env.OPEN_AI_API);

    const text = data[data.length - 1].content;
    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float'
    });

    const results = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.data[0].embedding
    });

    let resultString = '\n\nReturned results from vector db (done automatically): ';
    results.matches.forEach((match) => {
        resultString += `
        Professor: ${match.id},
        Review: ${match.metadata.review}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n
        `;
    });

    const lastMessage = data[data.length - 1];
    const lastMessageContent = lastMessage.content + resultString;
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1);
    const completion = await openai.chat.completions.create({
        messages: [
            { role: 'system', content: systemPrompt },
            ...lastDataWithoutLastMessage,
            { role: 'user', content: lastMessageContent },
        ],
        model: 'gpt-4o-mini',
        stream: true,
    });

    const stream = new ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder();
            try {
                for await (const chunk of completion) {
                    const content = chunk.choices[0]?.delta?.content;
                    if (content) {
                        const text = encoder.encode(content);
                        controller.enqueue(text);
                    }
                }
            } catch (err) {
                controller.error(err);
            } finally {
                controller.close();
            }
        }
    });
    console.log(stream)
    return new NextResponse(stream, {
        headers: {
            'Content-Type': 'text/plain'
        }
    });
}