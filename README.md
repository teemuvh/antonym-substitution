# It Is Not Easy To Detect Paraphrases

The *SemAntoNeg version 1.0* challenge set is provided in the testsuite directory.

The challenge set probes language models' representations of negation and opposing concepts in terms of antonymy.

In this challenge set, a model is supposed to choose the correct sentence for some input sentence from an option of three similar sentences. 
For instance, having an input of: *He's asleep.* the model should return the sentence *He's not awake.* from the options of *He's not asleep.*, *He's awake.* and *He's not awake.*

Thus, to succeed in this task, a language model needs to understand that insertion or deletion of a negation accompanied with antonym substitution produces a sentence 
that is semantically equivalent to the original sentence, and the sentence embeddings should represent this relationship.

The data is presented in the paper *It Is Not Easy To Detect Paraphrases: Analysing Semantic Similarity With Antonyms and Negation Using the New SemAntoNeg Benchmark*.
