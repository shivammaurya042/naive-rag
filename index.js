import dotenv from 'dotenv';
dotenv.config();

// import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import readline from 'readline';
import { stdin, stdout } from 'process';
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";
import { retriver } from './utils/retriever.js';

const rl = readline.createInterface({
    input: stdin,
    output: stdout
  });

try {
    const llm = new ChatOpenAI({ modelName: 'gpt-4o-mini' });

    // Create a prompt template for standalone question.
    const stdQTemplate = `given some conversation history (if any) and a question, create a standalone question from the following question.
    conversation history: {conv_history}
    question: {question}`;
    const stdQPrompt = PromptTemplate.fromTemplate(stdQTemplate);

    // Create a prompt template for llm answer.
    const llmAnswerTemplate =
        `you are an helpful bot that is meant to answer user queries from the given context.
        remember to be friendly. only answer from the context provided, if not found then use conversation history to get the answer. dont try to make up an answer.
        dont be lazy to provide complete answer. apologise if you don't know the answer and advise the user to email to shivammaurya042@gmail.com
        context: {context}
        conversation history: {conv_history}
        question: {question}`;
    const llmAnswerPrompt = PromptTemplate.fromTemplate(llmAnswerTemplate);

    const stdQChain = RunnableSequence.from([stdQPrompt, llm, new StringOutputParser()]);
    const retrieverChain = RunnableSequence.from([prev => prev.std_question, retriver, combinePageContent]);
    const llmAnswerChain = RunnableSequence.from([llmAnswerPrompt, llm, new StringOutputParser()]);

    // Create a chain for the question.
    // const chain = stdQPrompt.pipe(llm).pipe(new StringOutputParser()).pipe(retriver).pipe(combinePageContent()).pipe(llmAnswerPrompt);
    const chain = RunnableSequence.from([
        {
            std_question: stdQChain,
            original_input: new RunnablePassthrough()
        },
        {
            context: retrieverChain,
            question: ({original_input})=>original_input.question,
            conv_history: ({original_input})=>original_input.conv_history
        },
        llmAnswerChain]);

    const convHistory = [];
    while (true) {
        const question = await userQuestion('User turn: ');
        const response = await chain.invoke({ question: question, conv_history: formatConvHistory(convHistory) });

        convHistory.push(question, response);
        console.log('AI response: ', response);
    }

} catch (error) {
    console.log(error);
}

function combinePageContent(docs) {
    return docs.map(doc => doc.pageContent).join('\n\n');
}

function userQuestion(quest) {
    return new Promise(resolve => rl.question(quest, resolve));
}

function formatConvHistory(conv) {
    const formattedHistory = conv.map((message, i) => 
        i % 2 === 0 ? `user: ${message}` : `assistant: ${message}`
    ).join('\n');
    return formattedHistory;
}

// // below code is for embeddings and storing to vectore store.

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
