from collections import defaultdict
import cPickle
import os

import jieba
from base import *

__all__ = ['GroceryTextConverter']


def _dict2list(d):
    if len(d) == 0:
        return []
    m = max(v for k, v in d.iteritems())
    ret = [''] * (m + 1)
    for k, v in d.iteritems():
        ret[v] = k
    return ret


def _list2dict(l):
    return dict((v, k) for k, v in enumerate(l))


class GroceryTextPreProcessor(object):
    def __init__(self):
        # index must start from 1
        self.tok2idx = {'>>dummy<<': 0}
        self.idx2tok = None

    @staticmethod
    def _default_tokenize(text):
        #return jieba.cut(text.strip(), cut_all=True)
        return jieba.cut(text.strip(), cut_all=False)

    def purification(self,text,remove_stopwords=False):
    	text = BeautifulSoup(text).get_text()
	text = re.sub("[a-zA-Z]","",text)
	if remove_stopwords:
		pass
	return text

    def preprocess(self, text, custom_tokenize):
    	text = self.purification(text)
        if custom_tokenize is not None:
            tokens = custom_tokenize(text)
        else:
            tokens = self._default_tokenize(text)
	ret = ' '.join(tokens)
        return ret

    def save(self, dest_file):
        self.idx2tok = _dict2list(self.tok2idx)
        config = {'idx2tok': self.idx2tok}
        cPickle.dump(config, open(dest_file, 'wb'), -1)

    def load(self, src_file):
        config = cPickle.load(open(src_file, 'rb'))
        self.idx2tok = config['idx2tok']
        self.tok2idx = _list2dict(self.idx2tok)
        return self


class GroceryFeatureGenerator(object):
    def __init__(self):
	self.stop_words = None
	self.tfidfpath = 'resource/tf_idf.pickle'
	self.tfidf = TfidVectorizer(max_features=4000,
		ngram_range(1,3),sublinear_tf = True)
    def settfidf(self,stopwords_path=None,max_features=4000):
        if  stop_words != None:
		words = self.get_stopwords(stopwords_path)
    		self.ifidf.set_params(stop_words=words)
	if max_features != 4000:
    		self.ifidf.set_params(max_features=max_features)
	return self
    def get_stopwords(self,stop_words_path):
    	try:
	    with open(stop_words_path,'r') as fin:
	        contents = fin.read().decode('utf-8')
	 except IOError:
	 	raise ValueError("the given stop words path  is in invalid.")
	for line in contents.splitlines():
		self.stop_words.add(line.strip())
    def fit_transform(self,textlist):
    	if self.tiidf is None:
		self.load_tfidf()
    	if isinstance(textlist,list):
	    tf = self.tfidf.fit_transform(textlist)
	    with open(self.tfidfpath,'w') as fout:
	        pickle.dump(tf)
	    return tf
    def load_tfidf(self):
        with open(self.tfidfpath,'r') as fin:
	    self.tfidf = pickle.load(fin)

	    		
    def transform(self,textlist):
        if ininstance(textlist,list):
	    return self.tfidf.transform(textlist)

    	


class GroceryClassMapping(object):
    def __init__(self):
        self.class2idx = {}
        self.idx2class = None

    def to_idx(self, class_name):
        if class_name in self.class2idx:
            return self.class2idx[class_name]

        m = len(self.class2idx)
        self.class2idx[class_name] = m
        return m

    def to_class_name(self, idx):
        if self.idx2class is None:
            self.idx2class = _dict2list(self.class2idx)
        if idx == -1:
            return "**not in training**"
        if idx >= len(self.idx2class):
            raise KeyError(
                'class idx ({0}) should be less than the number of classes ({0}).'.format(idx, len(self.idx2class)))
        return self.idx2class[idx]

    def save(self, dest_file):
        self.idx2class = _dict2list(self.class2idx)
        config = {'idx2class': self.idx2class}
        cPickle.dump(config, open(dest_file, 'wb'), -1)

    def load(self, src_file):
        config = cPickle.load(open(src_file, 'rb'))
        self.idx2class = config['idx2class']
        self.class2idx = _list2dict(self.idx2class)
        return self


class GroceryTextConverter(object):
    def __init__(self, custom_tokenize=None):
        self.text_prep = GroceryTextPreProcessor()
        self.feat_gen = GroceryFeatureGenerator()
        self.class_map = GroceryClassMapping()
        self.custom_tokenize = custom_tokenize

    def get_class_idx(self, class_name):
        return self.class_map.to_idx(class_name)

    def get_class_name(self, class_idx):
        return self.class_map.to_class_name(class_idx)

    def to_svm(self, text, class_name=None):
        feat = self.feat_gen.bigram(self.text_prep.preprocess(text, self.custom_tokenize))
        if class_name is None:
            return feat
        return feat, self.class_map.to_idx(class_name)

    def convert_text(self, text_src, delimiter, output=None):
        if not output:
            output = '%s.svm' % text_src
        text_src = read_text_src(text_src, delimiter)
        with open(output, 'w') as w:
            for line in text_src:
                try:
                    label, text = line
                except ValueError:
                    continue
                feat, label = self.to_svm(text, label)
                w.write('%s %s\n' % (label, ''.join(' {0}:{1}'.format(f, feat[f]) for f in sorted(feat))))

    def save(self, dest_dir):
        config = {
            'text_prep': 'text_prep.config.pickle',
            'feat_gen': 'feat_gen.config.pickle',
            'class_map': 'class_map.config.pickle',
        }
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        self.text_prep.save(os.path.join(dest_dir, config['text_prep']))
        self.feat_gen.save(os.path.join(dest_dir, config['feat_gen']))
        self.class_map.save(os.path.join(dest_dir, config['class_map']))

    def load(self, src_dir):
        config = {
            'text_prep': 'text_prep.config.pickle',
            'feat_gen': 'feat_gen.config.pickle',
            'class_map': 'class_map.config.pickle',
        }
        self.text_prep.load(os.path.join(src_dir, config['text_prep']))
        self.feat_gen.load(os.path.join(src_dir, config['feat_gen']))
        self.class_map.load(os.path.join(src_dir, config['class_map']))
        return self
