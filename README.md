# Game Of Thrones Generative ChatBot(GOT-BOT)

Game Of Thrones ChatBot Trained using Seq2Seq Model on Game Of Thrones Subs and Book data

#### Dependencies
* [Python 2.7](https://www.python.org)
* [Tensorflow > 1.1](https://www.tensorflow.org/)
* [nltk](https://pypi.python.org/pypi/nltk)

## Pretrained Model
* Download Link: https://www.dropbox.com/s/o28ep1sachep2gc/checkpoints_got_bot.zip?dl=0

## Model Information
* RNN Encoder-Decoder Model(Seq2Seq)
* MultiRNNCell GRU Cell
* Attention Decoder and Embedding Layer with Buckets
* Sampled SoftMax Loss and SGD Optimizer
* Greedy Response Construction

## Dataset Preparation

The Data has been prepared from Game Of Thrones Subtitles(all episodes, https://www.opensubtitles.org) and Game Of Thrones Book(https://archive.org), All pre-processing and post-processing steps are available in Data_Prep file.

GOT SUBS: https://www.opensubtitles.org/en/ssearch/sublanguageid-all/idmovie-63130
GOT-BOOK: https://archive.org/details/A_Game_of_Thrones_Series

It can be used to train on any set of subtitles within the same file.

## Test Run
### Due to less amount of data, reponses are sometimes too vague. 

python Got_Bot.py --m=chat
Initialize new model
Create placeholders
Inference, SampleSoftmax
Creating loss...
Time: 3.61780786514
Create optimizer...
Loading parameters for the Chatbot
After Output--------------------------------
GOT-BOT: Message Limit: 16
--> Who are you
Output Logits:  [[-0.30675367  6.68275833 -0.25434929 ...,  0.01963711 -0.30866829
  -0.34872866]]
Output Greedy:  [151, 4, 1737, 5, 215, 4, 3, 4, 3, 22, 144, 4, 3, 3, 4, 3, 3, 4, 3]
Lannister . Tarth the Yes .
--> Who rules kings landing
Output Logits:  [[  0.45411149  15.40985966  -0.24345914 ...,  -0.08361083   0.32688218
    0.35243195]]
Output Greedy:  [2989, 83, 27, 11, 1232, 450, 3, 7, 590, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
Valar men with a Moon Kingdoms
--> where is my army of unsullied
Output Logits:  [[-0.78256309  4.53607988 -0.84746158 ..., -0.29136604 -0.5404861
  -0.06462321]]
Output Greedy:  [13, 8, 36, 83, 13, 35, 14, 748, 39, 5, 2787, 4, 3, 3, 4, 3, 3, 4, 3]
i I have men i He s standing on the battlefield .
--> 'The man who passes the sentence should swing the sword.
Output Logits:  [[  0.45567238  11.5950985   -1.02363682 ...,  -0.29802227  -0.66476578
    0.07089426]]
Output Greedy:  [378, 15, 340, 4, 3, 109, 15, 300, 4, 3, 3, 27, 1731, 4, 3, 4, 3, 3, 3]
under his side .
--> The things I do for love.
Output Logits:  [[-1.81315196  7.67702293 -1.41315424 ..., -0.63991416 -1.00333619
  -1.17256224]]
Output Greedy:  [5, 130, 10, 193, 7, 52, 5, 70, 59, 1, 3, 7, 5, 107, 1, 14, 289, 4, 3]
the way you So to do the man Lord <unk>
--> It's the family name that lives on. It's all that lives on.
Output Logits:  [[-0.94194341 -1.72485757 -0.64410913 ...,  0.37948987 -0.76213503
   0.36827829]]
Output Greedy:  [48, 469, 14, 2540, 1, 45, 1488, 560, 560, 19, 1102, 778, 109, 5, 537, 4, 3, 87, 14]
And someone s proved <unk> they offered which which he commands .. by the air .
--> I learned how to die a long time ago.
Output Logits:  [[ 0.43588781  1.38688314 -0.53811967 ...,  0.85255432 -0.06748818
   0.17954454]]
Output Greedy:  [13, 24, 137, 28, 27, 5, 1, 13, 13, 181, 29, 4534, 4, 3, 3, 3, 3, 13, 3]
i You come him with the <unk> i i This is bleed .
--> Turn us away, and we will burn you first.
Output Logits:  [[ -0.60744536  12.11123943  -0.68960094 ...,  -0.04841948  -0.99147111
   -0.6090138 ]]
Output Greedy:  [1523, 6, 34, 126, 41, 3, 4, 3, 80, 11, 1, 3, 4, 3, 3, 3, 7, 3, 3]
Enough , my lord she
--> A girl gives a man his own name?
Output Logits:  [[-0.22489437  3.75130343  0.36403471 ...,  1.052912   -0.42520836
   0.18865156]]
Output Greedy:  [9, 1641, 10, 228, 238, 56, 3, 6, 3, 91, 22, 3, 4, 3, 3, 4, 3, 3, 4]
and surely you might keep ?
--> A dragon is not a slave.
Output Logits:  [[ 0.76116204  5.55503654 -0.32142043 ..., -0.41793767  0.18629865
   0.2971383 ]]
Output Greedy:  [378, 72, 124, 177, 4, 3, 4, 3, 22, 177, 4, 3, 3, 27, 22, 177, 4, 3, 3]
under their good head .
--> The Lannisters send their regards 
Output Logits:  [[  0.29623079  11.3139143   -0.40246141 ...,  -0.67723441  -0.5055514
   -0.78701621]]
Output Greedy:  [5, 117, 6, 1, 184, 4, 3, 26, 11, 100, 3, 3, 4, 3, 3, 3, 3, 3, 3]
the only , <unk> again .
--> 

