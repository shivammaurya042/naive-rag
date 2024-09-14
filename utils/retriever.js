/* eslint-disable no-undef */
import { OpenAIEmbeddings } from "@langchain/openai";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { createClient } from '@supabase/supabase-js'

import dotenv from 'dotenv';
dotenv.config();

const sbapikey = process.env.SUPABASE_API_KEY;
const sburl = process.env.SUPABASE_URL_LC_CHATBOT;
const openAIApiKey = process.env.OPENAI_API_KEY;

const sbClient = createClient(sburl, sbapikey);
const embeddings = new OpenAIEmbeddings({ openAIApiKey })

// Vector store integration.
const vectoreStore = new SupabaseVectorStore(embeddings, {
    client: sbClient,
    tableName: 'documents',
    queryName: 'match_documents'
});

// Vectorstores can be converted to the retriever interface like:
export const retriver = vectoreStore.asRetriever();