/* eslint-disable no-unused-vars */
import dotenv from 'dotenv';
dotenv.config();

// import fs from 'fs/promises';
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
// import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { retriver } from './utils/retriever.js';

try {
    const llm = new ChatOpenAI({ modelName: 'gpt-4o-mini' });

    // Create a prompt template for the question.
    const stdQTemplate = 'create a standalone question from the following context: {context}';
    const stdQPrompt = PromptTemplate.fromTemplate(stdQTemplate);

    // Create a chain for the question.
    const stdQChain = stdQPrompt.pipe(llm).pipe(new StringOutputParser()).pipe(retriver);

    // Create a question.
    const question = "I am trying to understand Langchain. I wonder what are chains here.";

    // Invoke the chain.
    const responseFromVectorStore = await stdQChain.invoke({ context: question });

    let pageContent = '';
    for (const doc of responseFromVectorStore) {
        pageContent += doc.pageContent;
    }
    console.log(pageContent);

    // Create a prompt template for the answer.
    const llmAnswerTemplate = 
        `You are an helpful bot that is meant to answer user queries.
        Remember to be friendly. Only answer from the context provided. never makeup answers.
        apologise if you don't know the answer and advise the user to email to shivammaurya042@gmail.com
        context: {context}. question: {question}`;

    const llmAnswerPrompt = PromptTemplate.fromTemplate(llmAnswerTemplate);
    const llmAnswerChain = llmAnswerPrompt.pipe(llm);

    const llmResponse = await llmAnswerChain.invoke({ question: question, context: pageContent });
    console.log(llmResponse);

} catch (error) {
    console.log(error);
}

