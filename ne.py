import nltk
import sklearn
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import gensim
import re
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--dev', dest='dev', action='store_true', help = "Flag that determines whether to use training and dev set together or not.")
parser.add_argument('--cv', dest='cv', action='store_true', help = "Flag that determines whether to do cross validation.")
parser.add_argument('--details', dest='details', action='store_true', help = "Flag that determines whether to show details of mismatches.")

args = parser.parse_args()

######## READ INPUT DATA #########
def read_file(file_name):
    with open(file_name) as f:
        data = []
        data_temp = []
        labels = []
        labels_temp = []

        for line in f.read().splitlines():
            if line != '':
                a = line.split('\t')
                data_temp.append(a[0])
                labels_temp.append(a[1])
            else:
                data.append(data_temp)
                labels.append(labels_temp)
                data_temp = []
                labels_temp = []

    f.close()
    return data, labels

def read_no_labels(test_file_name):
    with open(test_file_name) as f:
        data = []
        data_temp = []

        for line in f.read().splitlines():
            if line != '':
                data_temp.append(line)
            else:
                data.append(data_temp)
                data_temp = []
    f.close()
    return data

# Read input data
train_data, train_labels = read_file('data/train/train.txt')
dev_data, dev_labels = read_file('data/dev/dev.txt')
test_data = read_no_labels('data/test/test.nolabels.txt')

######## READ LIST OF NAMES AND ENTITIES #########
with open("data/train/eng.list", encoding='utf-8') as f:
    eng_data = []
    for line in f.read().splitlines():
        if line != '':
            temp = line.split(' ')
            eng_data.append(temp[1:])
f.close()

def read_names():
    name_data = set([])

    def read_names_files(filename):
        with open(filename, encoding='utf-8') as f:
            #f.read() # Get rid of the header
            for line in f.read().splitlines():
                if line != '':
                    temp = line.split(',')
                    name_data.add(temp[0].strip(' \t\n\r').split(' ')[0])
                    name_data.add(temp[1].strip(' \t\n\r').split(' ')[0])
        f.close()

    read_names_files("male-names.csv")
    read_names_files("female-names.csv")
    return name_data

name_data = read_names()

######## LOAD WORD2VEC MODEL #########
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

# If the word doesn't exist, we return an array with NULL values.
def get_word_vec(word):
    try:
        return model[word]
    except KeyError:
        return ['NULL']*300

######## POS AND CHUNK PARSING #########
grammar = r"""
  NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
  PP: {<IN>}               # Chunk prepositions followed by NP
  VP: {<VB.*>} # Chunk verbs and their arguments
  """
cp=nltk.RegexpParser(grammar)

def chunck_tag(sentence):
    tree = cp.parse(sentence)
    return nltk.chunk.tree2conlltags(tree)

######## WORD FEATURES #########
init_words = [item[0] for item in eng_data]
middle_words = [item for sublist in eng_data for item in sublist[1:]]

lower_init_words = [item[0].lower() for item in eng_data]
lower_middle_words = [item.lower() for sublist in eng_data for item in sublist[1:]]
name_data = read_names()

def Word2Features(sentence, pos):
    features = {}
    features.update(current_word_features(sentence[pos][0]))
    features.update(w2vfeatuers(sentence[pos][0]))
    if pos > 0:
        features.update(prev_word_features(sentence[pos-1][0]))
#         features.update(w2vfeatuers(sentence[pos-1][0], suffix="prev_"))
        features.update(external_sources_features(sentence,pos-1, suffix="prev_"))
    else:
        features.update(begin_of_sentence())

    if pos < len(sentence)-1:
        features.update(next_word_features(sentence[pos+1][0]))
#         features.update(w2vfeatuers(sentence[pos+1][0], suffix="next_"))
        features.update(external_sources_features(sentence,pos+1, suffix="next_"))
    else:
        features.update(end_of_sentence())

    features.update(external_sources_features(sentence,pos))
    features.update(tag_features(sentence,pos))
    features.update(chunk_features(sentence,pos))
    return features


def w2vfeatuers(word, suffix=""):
    w2vfeatures = {}
    for index, letter in enumerate(get_word_vec(word)):
        w2vfeatures.update({suffix + 'wv_value'+str(index): letter})
    return w2vfeatures


def current_word_features(word):
    return {
        'bias': 1.0,
        'lower': word.lower(),
        'suffix_4': word[-4:],
        'suffix_3': word[-3:],
        'suffix_2': word[-2:],
        'isupper': word.isupper() * 25.0,
        'istitle': word.istitle() * 5.0,
        'isdigit': word.isdigit(),
#         'has_digit': True if re.match(r'.*[0-9].*', word) else False,
#         'single_digit': True if re.match(r'[0-9]', word) else False,
#         'double_digit': True if re.match(r'[0-9][0-9]', word) else False,
#         'has_dash': True if re.match(r'.*-.*', word) else False,
        'punct': True if re.match(r'[.,;:?!-+\'"]', word) else False,
        'istwittertag': isTwitterTag(word)
    }

def isTwitterTag(word):
    return word[0] == '@' or word[0] == '#'


def prev_word_features(word):
    return {
        'prev_lower': word.lower(),
        'prev_istitle': word.istitle(),
        'prev_isupper': word.isupper(),
    }


def next_word_features(word):
    return {
        'next_lower': word.lower(),
        'next_istitle': word.istitle(),
        'next_isupper': word.isupper(),
    }


def tag_features(sentence, pos):
    pos_features = {'pos[0]': sentence[pos][1]}
    prev_prev_pos_tag = sentence[pos-2][1] if pos > 1 else 'START'
    prev_pos_tag = sentence[pos-1][1] if pos > 0 else 'START'
    next_pos_tag = sentence[pos+1][1] if pos < len(sentence)-1 else 'END'
    next_next_pos_tag = sentence[pos+2][1] if pos < len(sentence)-2 else 'END'
    pos_features.update({
#         'pos[-2]': prev_prev_pos_tag,
#         'pos[-1]': prev_pos_tag,
#         'pos[+1]': next_pos_tag,
#         'pos[+2]': next_next_pos_tag,
        'pos[-2]|pos[-1]': prev_prev_pos_tag + '|' + prev_pos_tag,
        'pos[-1]|pos[0]': prev_pos_tag + '|' + sentence[pos][1],
        'pos[0]|pos[+1]': sentence[pos][1] + '|' + next_pos_tag,
        'pos[+1]|pos[+2]': next_pos_tag + '|' + next_next_pos_tag,
        'pos[-2]|pos[-1]|pos[0]': prev_prev_pos_tag + '|' + prev_pos_tag + '|' + sentence[pos][1],
        'pos[-1]|pos[0]|pos[+1]': prev_pos_tag + '|' + sentence[pos][1] + '|' + next_pos_tag,
        'pos[0]|pos[+1]|pos[+2]': sentence[pos][1] + '|' + next_pos_tag + '|' + next_next_pos_tag,
    })
    return pos_features


def chunk_features(sentence, pos):
    chunk_features = {'chunk[0]': sentence[pos][2]}
    prev_prev_chunk_tag = sentence[pos-2][2] if pos > 1 else 'NULL'
    prev_chunk_tag = sentence[pos-1][2] if pos > 0 else 'NULL'
    next_chunk_tag = sentence[pos+1][2] if pos < len(sentence)-1 else 'NULL'
    next_next_chunk_tag = sentence[pos+2][2] if pos < len(sentence)-2 else 'NULL'
    chunk_features.update({
#         'chunk[-1]': prev_chunk_tag,
#         'chunk[+1]': next_chunk_tag,
        'chunk[-2]|chunk[-1]': prev_prev_chunk_tag + '|' + prev_chunk_tag,
        'chunk[-1]|chunk[0]': prev_chunk_tag + '|' + sentence[pos][2],
        'chunk[0]|chunk[+1]': sentence[pos][2] + '|' + next_chunk_tag,
        'chunk[+1]|chunk[+2]': next_chunk_tag + '|' + next_next_chunk_tag,
#         'chunk[-2]|chunk[-1]|chunk[0]': prev_prev_chunk_tag + '|' + prev_chunk_tag + '|' + sentence[pos][2],
#         'chunk[-1]|chunk[0]|chunk[+1]': prev_chunk_tag + '|' + sentence[pos][2] + '|' + next_chunk_tag,
#         'chunk[0]|chunk[+1]|chunk[+2]': sentence[pos][2] + '|' + next_chunk_tag + '|' + next_next_chunk_tag,

    })
    return chunk_features


def begin_of_sentence():
    return {'BOS': True}


def end_of_sentence():
    return {'EOS': True}


def external_sources_features(sentence, pos, suffix=""):
    return {
        suffix + 'begin': is_begin_of_external(sentence[pos][0]),
        suffix + 'middle': is_middle_of_external(sentence[pos][0]),
        suffix + 'beginT': is_title_begin_of_external(sentence[pos][0]),
        suffix + 'middleT': is_title_middle_of_external(sentence[pos][0]),
        suffix + 'beginL': is_lower_begin_of_external(sentence[pos][0]),
        suffix + 'middleL': is_lower_middle_of_external(sentence[pos][0]),
        suffix + 'both': is_both_of_external(sentence[pos][0]),
        suffix + 'name': is_external_name(sentence[pos][0]),
    }

def is_begin_of_external(word):
    return word in init_words

def is_middle_of_external(word):
    return word in middle_words

def is_lower_begin_of_external(word):
    return word.lower() in lower_init_words

def is_lower_middle_of_external(word):
    return word.lower() in lower_middle_words

def is_title_begin_of_external(word):
    return (word.title() in init_words) * 4.0

def is_title_middle_of_external(word):
    return (word.title() in middle_words) * 4.0

def is_both_of_external(word):
    return is_begin_of_external(word) and is_middle_of_external(word)

def is_external_name(word):
    return (word.lower() in name_data) * 15.0

def is_any_external(word):
    return is_lower_begin_of_external(word) or is_lower_middle_of_external(word)

######## TRANSFORM INPUT #########
if args.dev:
    # Development training set
    train_data = [chunck_tag(nltk.pos_tag(train_data[i])) for i in range(len(train_data))]
    dev_data = [chunck_tag(nltk.pos_tag(dev_data[i])) for i in range(len(dev_data))]
    test_data = [chunck_tag(nltk.pos_tag(test_data[i])) for i in range(len(test_data))]

    X_train = [[Word2Features(s, pos) for pos in range(len(s))] for s in train_data]
    y_train = train_labels

    X_dev = [[Word2Features(s, pos) for pos in range(len(s))] for s in dev_data]
    y_dev = dev_labels
else:
    new_train_data = train_data + dev_data
    new_train_labels = train_labels + dev_labels
    new_train_data = [chunck_tag(nltk.pos_tag(new_train_data[i])) for i in range(len(new_train_data))]
    test_data = [chunck_tag(nltk.pos_tag(test_data[i])) for i in range(len(test_data))]
    X_train = [[Word2Features(s, pos) for pos in range(len(s)) ] for s in new_train_data]
    y_train = new_train_labels

X_test = [[Word2Features(s, pos) for pos in range(len(s))] for s in test_data]

######## HYPERPARAMETER TUNNING #########
if args.cv and args.dev:
    from sklearn.metrics import make_scorer
    from sklearn.cross_validation import cross_val_score
    from sklearn.grid_search import GridSearchCV

    # define fixed parameters and parameters to search
    crf_CV = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )
    params_space = {
        'c1': [0.5, 0.6, 0.65, 0.7, 0.75, 0.8],
        'c2': [0.5, 0.6, 0.65, 0.7, 0.75, 0.8],
    }

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='weighted', labels=labels)

    # search
    rs = GridSearchCV(crf_CV, params_space,
                            cv=3,
                            verbose=1,
                            n_jobs=-1,
                            scoring=f1_scorer)
    rs.fit(X_train, y_train)

    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)
    print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

    print("This are the results of all training")
    for score in rs.grid_scores_:
        print(score)
    quit()

######## TRAINING #########
crf_final = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.5,
    c2=0.5,
    max_iterations=100,
    all_possible_transitions=True
)
crf_final.fit(X_train, y_train)

# Just keep the 'B' and 'I' for F-1 scoring
labels = list(crf_final.classes_)
labels.remove('O')

######## CLASSIFICATION #########
print("Running on the training set")
y_train_pred = crf_final.predict(X_train)
print("F1-score" + str(metrics.flat_f1_score(y_train, y_train_pred, average='weighted', labels=labels)))

if args.dev:
    print("Running on the dev set")
    y_dev_pred = crf_final.predict(X_dev)
    print("F1-score" + str(metrics.flat_f1_score(y_dev, y_dev_pred,average='weighted', labels=labels)))

print("Running on the testing set")
y_test_pred = crf_final.predict(X_test)

######## OUTPUT #########
def generate_output(pred, outputfile):
    f = open(outputfile,'w')
    for label_sentence in pred:
        for label_word in label_sentence:
            f.write(label_word + '\n')
        f.write("\n")
    f.close()

print("Generating output file (output-train) for training set")
generate_output(y_train_pred, "output-train")
if args.dev:
    print("Generating output file (output-dev) for dev set")
    generate_output(y_dev_pred, "output-dev")
print("Generating output file (output-test) for testing set")
generate_output(y_test_pred, "output-test")

if args.dev and args.details:
    print("Generating list of mismatches")
    print(metrics.flat_classification_report(y_dev, y_dev_pred, labels=labels, digits=3))

    print("Generating list of mismatches")
    for i, sentence in enumerate(dev_data):
        if y_dev_pred[i] != dev_labels[i]:
            print('\n')
            print(sentence)
        for j in range(len(sentence)):
            if y_dev_pred[i][j] != dev_labels[i][j]:
                print(str(sentence[j]) + "is " + str(dev_labels[i][j]) + ", but we said " + str(y_dev_pred[i][j]) + \
                      "Name " + str(is_external_name(sentence[j][0])) + "External:" + str(is_any_external(sentence[j][0])))
