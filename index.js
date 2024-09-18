/* eslint-disable no-unused-vars */
import dotenv from 'dotenv';
dotenv.config();

// import fs from 'fs/promises';
// import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";
import { retriver } from './utils/retriever.js';

try {
    const llm = new ChatOpenAI({ modelName: 'gpt-4o-mini' });

    // Create a prompt template for standalone question.
    const stdQTemplate = 'create a standalone question from the following question: {question}';
    const stdQPrompt = PromptTemplate.fromTemplate(stdQTemplate);

    // Create a prompt template for the answer.
    const llmAnswerTemplate =
        `You are an helpful bot that is meant to answer user queries.
        Remember to be friendly. Only answer from the context provided. never makeup answers.
        apologise if you don't know the answer and advise the user to email to shivammaurya042@gmail.com
        context: {context}. question: {question}`;
    const llmAnswerPrompt = PromptTemplate.fromTemplate(llmAnswerTemplate);

    // user question.
    const question = "langhchain is great but I wonder what exactly is a chain here.";

    // Create a chain for the question.
    // const chain = stdQPrompt.pipe(llm).pipe(new StringOutputParser()).pipe(retriver).pipe(combinePageContent()).pipe(llmAnswerPrompt);

    const stdQChain = RunnableSequence.from([stdQPrompt, llm, new StringOutputParser()]);
    const retrieverChain = RunnableSequence.from([prev => prev.std_question, retriver, combinePageContent]);
    const llmAnswerChain = RunnableSequence.from([llmAnswerPrompt, llm, new StringOutputParser()]);

    const chain = RunnableSequence.from([
        {
            std_question: stdQChain,
            original_input: new RunnablePassthrough()
        },
        {
            context: retrieverChain,
            question: ({original_input})=>original_input.question
        },
        llmAnswerChain]);

    const response = await chain.invoke({ question: question });
    console.log('response', response);


    // Invoke the chain.
    // const responseFromVectorStore = await chain.invoke({ context: question });

    // const llmAnswerChain = llmAnswerPrompt.pipe(llm);

    // const responseFromVectorStore1 = await stdQChain.invoke({ context: question });


    // const llmResponse = await llmAnswerChain.invoke({ question: question, context: pageContent });


} catch (error) {
    console.log(error);
}

function combinePageContent(docs) {
    return docs.map(doc => doc.pageContent).join('\n\n');
}

// // get our docs and convert into embeddings to store in vectore store.

// try {
//     const text = await fs.readFile('text.txt', 'utf-8');
//     const splitter = new RecursiveCharacterTextSplitter({
//         chunkSize: 500,
//         separators: ['\n\n', '\n', ' ', ''],
//         chunkOverlap: 50
//     });

//     const outputSplits = await splitter.createDocuments([text]);

//     const sbapikey = process.env.SUPABASE_API_KEY;
//     const sburl = process.env.SUPABASE_URL_LC_CHATBOT;
//     const openAIApiKey = process.env.OPENAI_API_KEY;

//     const sbClient = createClient(sburl, sbapikey);

//     await SupabaseVectorStore.fromDocuments(
//         outputSplits,
//         new OpenAIEmbeddings({ openAIApiKey }),
//         {
//             client: sbClient,
//             tableName: 'documents',
//         });
// } catch (error) {
//     console.log(error)
// }
