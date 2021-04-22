# NLP_Project
Step:
1 . clone the branch .
2. Install the dependenciese using mentioned in the requirement.txt

Models Implemented :

1.Naive Bayes :
The only change you might need to make is in how the csv files are read into the program. Other than that as long as the correct
csv files are read the program should run just fine.


2:BiLSTM - based Model
  For this model we used SBERT vector embeddings as input . We tried with both "Input -> BiLSTM -> flatten -> dense"  and "Input -> BiLSTM -> BiLSTM-> flatten ->     dense.
  How to run.
  1) Prepare the vector embedding by running , command : python data_preprocessing/sentence_bert_embedding.py
  2) Run the run_BiLSTM.py , command : python model_architectures/run_BiLSTM.py
 
