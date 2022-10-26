# It Is Not Easy To Detect Paraphrases

The *SemAntoNeg version 1.0* challenge set is provided in the testsuite directory.

The challenge set probes language models' representations of negation and opposing concepts in terms of antonymy; 
a model is supposed to choose the correct sentence for some input sentence from an option of three similar sentences. 
For instance, having an input of: *He's asleep.* the model should return the sentence *He's not awake.* from the options of *He's not asleep.*, *He's awake.* and *He's not awake.*

To succeed in this task, a language model needs to recognise that antonym substitution requires either an insertion or a deletion of a negation marker to produce 
a sentence that is semantically equivalent to the original sentence, and the sentence embeddings should represent this relationship.

The data is introduced in the paper *It Is Not Easy To Detect Paraphrases: Analysing Semantic Similarity With Antonyms and Negation Using the New SemAntoNeg Benchmark*,
presented in the BlackboxNLP 2022 workshop, collocated with the EMNLP 2022.
