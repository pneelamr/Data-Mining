import pandas as pd
from pandas import DataFrame
import pickle

def get_text_features (text, n_feat, ng, isbinary):
    from sklearn.feature_extraction.text import TfidfVectorizer
    print ("inside get_text_features")
    print ("grams: " + str(ng))
    if isbinary:
        print ("Binary")
    # check if sub tf need pas for bin
    vectorizer = TfidfVectorizer(max_df=0.5, ngram_range = (ng,ng),sublinear_tf = True, max_features=n_feat, binary = isbinary)
    X = vectorizer.fit_transform(text)
    print("n_samples: %d, n_features: %d" % X.shape)
    feature_names = vectorizer.get_feature_names()
    return (X, feature_names)


def get_additional_features(additional):
    from sklearn.feature_extraction.text import CountVectorizer
    print ("inside get_additional_features")
    def get_bin(a, bin_width, start=0):
        return str(int((a - start) // bin_width))
    import re
    import json
    vectorizer = CountVectorizer()
    additional_feats = []
    PATTERN = re.compile(r'''((?:[^,"']|"[^"]*"|'[^']*')+)''')
    zipcodes = set([])
    token_dict = {}
    for i, line in enumerate(additional):
        ### FOR REPORT
            if i == 1:
                print ("FOR REPORT ADDITIONAL FEATURES BEFORE TOKENIZE")
                print line

            feats = PATTERN.split(line.strip())[1::2]
            zip = feats[1]
            locbin = get_bin(int(feats[2]), 10,start = 0)
            ratebin = get_bin(float(feats[3]),0.5,start=0)
            modified = "{ 'business':"+ (feats[0])[1:-1]+"}"
            modified = modified.replace("'", "\"")
            business_list = json.loads(modified)
            add_feats = []
            for business in business_list["business"]:
                business = business.replace(" ", "")
                if business != 'restaurants':
                    add_feats.append("cat"+business)
            add_feats.append("zip"+zip)
            add_feats.append("reviewbin"+locbin)
            add_feats.append("ratingbin"+ratebin)
            token_dict[i] = u' '.join(add_feats)
            ### FOR REPORT
            if i == 1:
                print ("FOR REPORT ADDITIONAL FEATURES AFTER TOKENIZE")
                print token_dict[i]

    X = vectorizer.fit_transform(token_dict.values())
    print("n_samples: %d, n_features: %d" % X.shape)
    feature_names = vectorizer.get_feature_names()
    return (X, feature_names)
