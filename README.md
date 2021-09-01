
# ISL

ISL is a project to enhance the communication capabilities of people with hearing abilities.

## Introduction
Sign language is a visual means of communicating through hand signals, gestures, facial expressions, and body language.
It’s the main form of communication for the Deaf and Hard-of-Hearing community, but sign language can be useful for other groups of people as well. 

ISL(Indian Sign Language) is predominantly used in Indian subcontinent. It is used by at least 7 million deaf signers.

The model takes a sentence(voice/text) as input and displays an ISL representative video of the sentence, keeping in mind all the rules and grammar of ISL. 

This model have been implemented by using two different parsers. ISL benepar.py uses benepar parser while model.py uses CoreNLPparser.

The goal is to help the deaf community by giving them resources by using which they don't feel inferior and get facilties so they can unleash their full potential and their disability be no more the obstacle between them and their dreams.

## Dataset

The videos of the dataset were downloaded from https://indiansignlanguage.org/ and then processed to make the final dataset.
Currently, our dataset consists of more than 300 words.

## Features

• It's first of it's kind which uses real persons instead of animations in displayed video.

• Can process voice both with internet and without internet.

• Uses synonyms of words, which are selected manually, so that if any word is used outside of dataset but has a word similar to it in our dataset, then it will be replaced by the word similar to it present in our dataset.

• If any word or a word similar to it is not present in our dataset, then it's letter by letter representation is displayed.

• The model keeps all rules of ISL grammar in mind while conversion of a sentence from english to it's ISL representation.

• Can handle short forms and contractions of words too.
## Examples

## Processing-

**Input mode-** Text

![Screenshot (441)](https://user-images.githubusercontent.com/65160713/131209520-70ec47d4-4b65-4aab-9dcf-12726d7abd0a.png)

**Input mode-** Audio

![Screenshot (442)](https://user-images.githubusercontent.com/65160713/131209745-fafbd756-0e90-4d7c-820d-deb49e058cd6.png)


**Input-** How are you

**ISL representation -** You how

**Output-**



https://user-images.githubusercontent.com/65160713/131209061-a49936f1-048e-4573-b564-587390893e5e.mp4


**Input-** "Why is your annual salary less"

**ISL representation** - "your year income less why"

**Note-** As we can see 'annual' is not presented in our dataset so it replaces it with 'yearly' which is then represented by 'year' in ISL.

**Output**- 



https://user-images.githubusercontent.com/65160713/131209231-b137fa45-7400-4716-8919-341c92c96d90.mp4




**Input**- "She's not gonna do this"

**ISL representation** - "She this not go to do"

**Note**- As we can see it handles contractions like 'she's' and 'gonna' where 'she's' is converted to 'she is' and 'gonna' to 'going to'.

**Output**- 


https://user-images.githubusercontent.com/65160713/131209017-a6bb4707-84d0-473d-8af8-6c9ca9a438e9.mp4



**Input**- "I will visit Kanpur next year"

**ISL representation** - "I kanpur will visit next year" 

**Output**-



https://user-images.githubusercontent.com/65160713/131209111-13f65d3b-d9a5-4cb0-93b6-002291362cd6.mp4



**Input**- "His leg was paining because he met with an accident yesterday"

**ISL representation**- "it yesterday his leg pain met with accident"

**Note**- As we can see it can handle big and complex sentences too.

**Output**-


https://user-images.githubusercontent.com/65160713/131209322-6d4bbe8b-6db4-4b18-9e7d-23547b760925.mp4


## Future plans

• To make dataset with 1-2 persons to maintain the consistency of output video.

• Make it's vice versa, i.e. indian sign language to text.

• To add SIGML animation too as an option

• Implement parallel processing so that as soon as someone keeps on speaking, it keeps on converting that to sign language sentence by sentence, until the person stops so that we can use it in real life like in schools and colleges, seminars, videos with deaf translation and many more...
