#!/usr/bin/env python

from ctypes import *
from ctypes.util import find_library
import sys
import os
from os import path
import shutil
from sklearn.linear_model import LogisticRegression as LR 
if sys.version_info[0] >= 3:
    xrange = range
    import pickle as cPickle
    izip = zip

    def unicode(string, setting):
        return string
else:
    import cPickle
    from itertools import izip

class LearnerModel:
    """
    :class:`LearnerModel` is a middle-level classification model. It
    inherits from :class:`liblinear.model` by having two more members:
    a :class:`LearnerParameter` instance and an inverse document frequency list.

    We do not recommend users to create a :class:`LearnerModel` by themselves.
    Instead, users should create and manipulate a :class:`LearnerModel`
    via :func:`train`, :func:`predict`, and :func:`predict_one`.

    If users want to redefine :class:`LearnerModel`, they must
    implement the following four methods used by
    :mod:`libshorttext.classifier` and :mod:`libshorttext.analyzer`.
    """


    def __init__(self, c_model, param = None, idf = None):
        """
        constructor of :class:`LearnerModel`.
        """
        self.c_model = c_model        
#        print_debug('c_model(%s), self(%s)' % (id(c_model), id(self)))
#
#        if isinstance(c_model, str):
#            self.load(c_model)
#            return
#        elif isinstance(c_model, liblinear.model):
#            if param is None:
#                raise ValueError("param can not be None if model is given.")
#        else:
#            raise TypeError("c_model should be model file name or a model.")
#
#        self.c_model = c_model # prevent GC
#
#        if isinstance(param, LearnerParameter):
#            self.param_options = param.raw_options
#        elif isinstance(param, tuple):
#            self.param_options = param
#        else:
#            raise TypeError("param should be a LearnerParameter or a tuple.")
#
#        if idf is not None:
#            self.idf = idf[:self.c_model.nr_feature + (self.c_model.bias >= 0)]
#        else:
#            self.idf = None
#
#        for attr in c_model._names:
#            setattr(self, attr, getattr(c_model, attr))
#
#        self._reconstruct_label_idx()



    def load(self, model_dir):
        """
        Load the contents from a :class:`TextModel` directory.
        """

        #self.c_model = liblinear_load_model(path.join(model_dir,'liblinear_model'))
        try:
            with open(path.join(model_dir,'LR.model'),'r') as fin:
                self.c_model = cPickle.load(fin)
        except IOError:
            raise ValueError('save model is fail.')
        return self
        #options_file = path.join(model_dir,'options.pickle')
        #self.param_options = cPickle.load(open(options_file,'rb'))

        #idf_file = path.join(model_dir,'idf.pickle')
        #self.idf = cPickle.load(open(idf_file,'rb'))

        #self.__init__(self.c_model, LearnerParameter(self.param_options[0], self.param_options[1]), self.idf)
    def save(self, model_dir, force=False):
        """
        Save the model to a directory. If *force* is set to ``True``,
        the existing directory will be overwritten; otherwise,
        :class:`IOError` will be raised.
        """

        if path.exists(model_dir):
            if force:
                shutil.rmtree(model_dir)
            else :
                raise OSError('Please use force option to overwrite the existing files.')
        os.mkdir(model_dir)
        try:
            modelfile = path.join(model_dir,'LR.model')
            with open(modelfile,'w') as fout:
                cPickle.dump(self.c_model,fout)
        except IOError:
            raise ValueError("the LR.model path is open fail")
        #liblinear_save_model(path.join(model_dir,'liblinear_model'), self.c_model)
        #options_file = path.join(model_dir,'options.pickle')
        #cPickle.dump(self.param_options, open(options_file,'wb'),-1)

        #idf_file = path.join(model_dir,'idf.pickle')
        #cPickle.dump(self.idf, open(idf_file,'wb'),-1)

#    def __str__(self):
#        if type(self.param_options) is tuple and len(self.param_options) > 0:
#            return 'LearnerModel: ' + (self.param_options[0] or 'default')
#        else:
#            return 'empty LearnerModel'


def train(data_in, learner_opts="", liblinear_opts=""):
    """
    Return a :class:`LearnerModel`.

    *data_file_name* is the file path of the LIBSVM-format data. *learner_opts* is a
    :class:`str`. Refer to :ref:`learner_param`. *liblinear_opts* is a :class:`str` of
    LIBLINEAR's parameters. Refer to LIBLINEAR's document.
    """

    #learner_prob = LearnerProblem(data_file_name)
    #learner_param = LearnerParameter(learner_opts, liblinear_opts)

    #idf = None
    #if learner_param.inverse_document_frequency:
    #    idf = learner_prob.compute_idf()

    #learner_prob.normalize(learner_param, idf)

    #m = liblinear_train(learner_prob, learner_param)
    if isinstance(data_in,str):
        pass
    if isinstance(data_in,list) or isinstance(data_in,tuple):
        train_x,label = data_in
    m = LR(penalty='l1')
#    if not learner_param.cross_validation:
#        m.x_space = None  # This is required to reduce the memory usage...
#        m = LearnerModel(m, learner_param, idf)
    m.fit(train_x,label)
    m = LearnerModel(m)
    return m


#def predict_one(xi, m):
def predict_one(xi,model):
    """
    Return the label and a :class:`c_double` array of decision values of
    the test instance *xi* using :class:`LearnerModel` *m*.

    *xi* can be a :class:`list` or a :class:`dict` as in LIBLINEAR python
    interface. It can also be a LIBLINEAR feature_node array.

    .. note::

        This function is designed to analyze the result of one instance.
        It has a severe efficiency issue and should be used only by
        :func:`libshorttext.classifier.predict_single_text`. If many
        instances need to be predicted, they should be stored in a file
        and predicted by :func:`predict`.

    .. warning::

        The content of *xi* may be **changed** after the function call.
    """

#    if isinstance(xi, (list, dict)):
#        xi = liblinear.gen_feature_nodearray(xi)[0]
#    elif not isinstance(xi, POINTER(liblinear.feature_node)):
#        raise TypeError("xi should be a test instance")
#
#    learner_param = LearnerParameter(m.param_options[0], m.param_options[1])
#
#    if m.bias >= 0:
#        i = 0
#        while xi[i].index != -1: i += 1
#
#        # Already has bias, or bias reserved.
#        # Actually this statement should be true if
#        # the data is read by read_SVMProblem.
#        if i > 0 and xi[i-1].index == m.nr_feature + 1:
#            i -= 1
#
#        xi[i] = liblinear.feature_node(m.nr_feature + 1, m.bias)
#        xi[i+1] = liblinear.feature_node(-1, 0)
#
#    LearnerProblem.normalize_one(xi, learner_param, m.idf)
#
#    dec_values = (c_double * m.nr_class)()
#    label = liblinear.liblinear.predict_values(m, xi, dec_values)
    if model in None:
        return
    pl = model.c.model.predict([xi])
    return pl


#def predict(data_file_name, m, liblinear_opts=""):
def predict(datain,model):
    """
    Return a quadruple: the predicted labels, the accuracy, the decision values, and the
    true labels in the test data file (obtained through the :class:`LearnerModel` *m*).

    The predicted labels and true labels in the file are :class:`list`. The accuracy is
    evaluated by assuming that the labels in the file are the true label.

    The decision values are in a :class:`list`, where the length is the same as the number
    of test instances. Each element in the list is a :class:`c_double` array, and the
    values in the array are an instance's decision values in different classes.
    For example, the decision value of instance i and class k can be obtained by

    >>> predicted_label, accuracy, all_dec_values, label = predict('svm_file', model)
    >>> print all_dec_values[i][k]
    """
    if isinstance(datain,str):
        pass
    if isinstance(datain,list):
        Xin = datain
#    learner_prob = LearnerProblem(data_file_name)
#    learner_param = LearnerParameter(m.param_options[0], m.param_options[1])
#
#    idf = None
#    if m.idf:
#        idf = (c_double * len(m.idf))()
#        for i in range(len(m.idf)): idf[i] = m.idf[i]
#    learner_prob.normalize(learner_param, idf)
#
#    all_dec_values = []
#    acc = 0
#    py = []  # predicted y
#    ty = []  # true y
#
#    dec_values = (c_double * m.nr_class)()
#
#    for i in range(learner_prob.l):
#        label = liblinear.liblinear.predict_values(m, learner_prob.x[i], dec_values)
#        all_dec_values += [dec_values[:m.nr_class]]
#        py += [label]
#        ty += [learner_prob.y[i]]
#
#        if label == learner_prob.y[i]:
#            acc += 1
#
#    acc /= float(learner_prob.l)
    pl = model.c_model.predict(datain)
    return pl
    #return py, acc, all_dec_values, ty



#if __name__ == '__main__':
#    argv = sys.argv
#    if len(argv) < 2: #4 or '-v' not in argv:
#        print("{0} -v fold [other liblinear_options] [learner_opts] training-data".format(argv[0]))
#        sys.exit(-1)
#    data_file_name = argv[-1]
#    learner_opts, liblinear_opts = [], []
#    i = 1
#    while i < len(argv)-1:
#        if argv[i] in ["-D", "-N", "-I", "-T"]:
#            learner_opts += [argv[i], argv[i+1]]
#            i += 2
#        else :
#            liblinear_opts += [argv[i]]
#            i += 1
#    m = train(data_file_name, learner_opts, liblinear_opts)
