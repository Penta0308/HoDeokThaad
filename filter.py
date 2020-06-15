from konlpy.tag import Hannanum
import nltk
import pickle

pos_tagger = Hannanum()

def save(classifier):
    save_classifier = open("radarconfig.pickle", "wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()

def load():
    classifier_f = open("radarconfig.pickle", "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier
# https://dbrang.tistory.com/1268

def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc)]

def term_exists(doc):
    return {'exists({})'.format(word): (word in set(doc)) for word in tokens}

train = [("카지노사이트", "neg"),
         ("아이 적성검사 받으셨나요?", "neg"),
         ("연세대 무료체험 떴네요~~ 아이들 있으면 마감전에 챙기세요!", "neg"),
         ("초등학생 자녀 있으신분들 무료로 다중지능검사 받아보세요!!", "neg"),
         ("일요일 일식있다네요", "pos"),
         ("KSP 포럼 회칙 개정/간결안 의견수렴", "pos"),
         ("문화어 패치 개발비화 (사진 용량 큼)", "pos"),
         ("카페 회칙 수정안 투표", "pos")]

train_docs=[]

for row in train:
    train_docs.append((tokenize(row[0]), row[1]))

tokens = [t for d in train_docs for t in d[0]]

train_xy = [(term_exists(d), c) for d, c in train_docs]

classifier = nltk.NaiveBayesClassifier.train(train_xy)

#classifier = load()

classifier.show_most_informative_features()

#save(classifier)