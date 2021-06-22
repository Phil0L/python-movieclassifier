import os
import pickle
import sqlite3
import time
from shutil import copyfile

import numpy as np

# import HashingVectorizer from local dir
from movieclassifier.vectorizer import vect


def update_model(db_path, model, batch_size=10000):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT * from review_db')

    results = c.fetchmany(batch_size)

    while results:
        data = np.array(results)
        X = data[:, 0]
        y = data[:, 1].astype(int)

        classes = np.array([0, 1])
        X_train = vect.transform(X)
        model.partial_fit(X_train, y, classes=classes)
        results = c.fetchmany(batch_size)

    conn.close()
    return model


cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,
                                    'pkl_objects',
                                    'classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'reviews.sqlite')
clf = update_model(db_path=db, model=clf, batch_size=10000)
# Uncomment the following line if you are sure that
# you want to update your classifier.pkl file
# permanently.
timestr = time.strftime("%Y%m%d-%H%M%S")
orig_path = os.path.join(cur_dir, 'pkl_objects', 'classifier.pkl')
backup_path = os.path.join(cur_dir, 'pkl_objects', 'classifier_%s.pkl' % timestr)
copyfile(orig_path, backup_path)
pickle.dump(clf, open(os.path.join(cur_dir, 'pkl_objects', 'classifier.pkl'), 'wb'), protocol=4)
