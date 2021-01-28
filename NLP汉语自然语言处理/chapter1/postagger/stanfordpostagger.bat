java -mx300m -cp "E:\\nltk_data\\stanfordpostagger\\stanford-postagger.jar;E:\\nltk_data\\stanfordpostagger\\lib\\slf4j-api.jar;E:\\nltk_data\\stanfordpostagger\\lib\\slf4j-simple.jar;" edu.stanford.nlp.tagger.maxent.MaxentTagger -model "E:\\nltk_data\\stanfordpostagger\\models\\chinese-distsim.tagger" -textFile postest.txt > result.txt

pause;