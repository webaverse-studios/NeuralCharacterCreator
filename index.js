// import dotenv
import dotenv from 'dotenv';
dotenv.config();
import '@tensorflow/tfjs-node';
import use from '@tensorflow-models/universal-sentence-encoder';
import { readFileSync } from 'fs';
import * as qna from '@tensorflow-models/qna';
import { Configuration, OpenAIApi } from "openai";
import minimist from "minimist";
import fs from "fs";

var argv = minimist(process.argv.slice(2));


const makePrompt = (name, description) =>
    `
Name: Annie Sayiu
Description: A very skilled sword dancer. 15/F. She loves to dance and is very passionate about it. She is also very deadly with her blades. She likes to wear lightweight clothing, a silk sash, a golden amulet and several bracelets on each wrist. She of course wields two swords when she fights.
What is Annie wearing?
body: Lightweight silk robe
chest: none
feet: Padded Slippers
hands: Bracelets
waist: Silk Sash
head: none
legs: none
weapon: Dancing Swords
neck: Amulet

Name: Scillia
Description: Drop hunter of unknown age, although she appears to be a 13 year old girl. Really she's an AI. She carries a big sword (called a Sil Sword). Her hair is white, but it  turns purple when she goes Super Sayin. She usually wears a gray coat with buttons on it, combat boots and a skirt so she can run really fast.
What is Scillia wearing?
body: Gray coat with buttons
chest: none
feet: Combat boots
hands: none
waist: none
head: none
legs: Skirt
weapon: Sil Sword
neck: none

Name: ${name}
Description: ${description}
What is ${name} wearing?
body:`

async function extractDescription(name, description) {

    const configuration = new Configuration({
        apiKey: process.env.OPENAI_API_KEY,
    });
    const openai = new OpenAIApi(configuration);

    const response = await openai.createCompletion({
        model: "text-davinci-002",
        prompt: makePrompt(name, description),
        temperature: 0.7,
        max_tokens: 256,
        top_p: 1,
        frequency_penalty: 0,
        presence_penalty: 0,
    });

    const data = 'body:' + response.data.choices[0].text;
    return data.split('\n');
}



// const qnaModel = await qna.load();

// read JSON from dataset.json
const json = JSON.parse(readFileSync('dataset.json', 'utf8'));

const traits = {};

// iterate through objects in JSON
for (const obj of json) {
    traits[obj.trait] = obj.collection.map(item => item.name + (item.description ? (' ' + item.description) : ''));
}

let traitEmbeddings = {};

// check if embeddings.json exists
// if it does, read it and set the resulting JSON to traitEmbeddings
if (fs.existsSync('embeddings.json')) {
    traitEmbeddings = JSON.parse(readFileSync('embeddings.json', 'utf8'));
} else {
    // for each trait in traits...
    for (const trait in traits) {
        const traitsArray = traits[trait];
        console.log('traitsArray', traitsArray);
        let embeddings = await get_embeddings(traitsArray);
        const embeds = embeddings.arraySync();
        traitEmbeddings[trait] = embeds;
    }
    // serialize traitEmbeddings to JSON and write it to embeddings.json
    fs.writeFileSync('embeddings.json', JSON.stringify(traitEmbeddings));
}



// get the first two args from the cli

if(!argv.name || !argv.description) {
    console.warn("Warning, you have called the script without passing a --name and --description argument. This will result in the default character being generated.");
    console.warn("Try `npm run start -- --name \"My Character\" --description \"My detailed character description");
}

const inputName = argv.name ?? "Drake";
const inputDescription = argv.description ?? argv.desc ?? "Neural Hacker. He is wearing motorcycle gloves, leather combat boots, a scarf that says 0xDEADBEEF, as well as a motorcycle jacket and black pants. He uses all kinds of guns, especially Uzis."

console.log('inputName', inputName);
console.log('inputDescription', inputDescription);

const desc = await extractDescription(inputDescription, inputName);

const outputTraits = {};

// search function to compare inputDescription to all descriptions in dataset and find an appropriate item for each "trait"
// trait examples: body, head, weapons, etc.

// for each trait..
// ask the question "what is the character wearing on their <trait>? None if none"

for (const trait in traits) {
    // get the array element in desc that contains the trait
    const traitDesc = desc.find(item => item.includes(trait));
    outputTraits[trait] = traitDesc.replace(`${trait}: `, '');
}

console.log('outputTraits', outputTraits);

const finalCharacter = {}

// for each trait in outputTraits, find the most similar item in the dataset
for (const trait of Object.keys(outputTraits)) {
    const e = await get_embeddings([outputTraits[trait]]);
    const traitEmbed = (e.arraySync())[0];

    let topSimilarityScore = 0;
    if (outputTraits[trait].includes('none')) {
        finalCharacter[trait] = null;
    }
    else {
        let embeds = traitEmbeddings[trait];

        for (let i = 0; i < traits[trait].length; i++) {
            const item = traits[trait][i];
            const itemEmbed = embeds[i];
            const similarity = cosineSimilarity(traitEmbed, itemEmbed);
            if (similarity > topSimilarityScore) {
                topSimilarityScore = similarity;
                const replaceString = trait + ': ';
                finalCharacter[trait] = item.replace(replaceString, '').trim();
            }
        }
    }

}

console.log('finalCharacter', finalCharacter);

// if response is none, calculate similarity between inputDescription and each name in trait

async function get_embeddings(list_sentences) {
    const model = await use.load();
    return await model.embed(list_sentences)
}

function dotProduct(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

function cosineSimilarity(a, b) {
    // https://towardsdatascience.com/how-to-build-a-textual-similarity-analysis-web-app-aa3139d4fb71

    const magnitudeA = Math.sqrt(dotProduct(a, a));
    const magnitudeB = Math.sqrt(dotProduct(b, b));
    if (magnitudeA && magnitudeB) {
        // https://towardsdatascience.com/how-to-measure-distances-in-machine-learning-13a396aa34ce
        return dotProduct(a, b) / (magnitudeA * magnitudeB);
    } else {
        return 0;
    }
}