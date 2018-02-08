import fasttext

def main():
    model = fasttext.skipgram('fasttext_train_full.txt', 'model_skipgram')
    model = fasttext.cbow('fasttext_train_full.txt', 'model_cbow')

if __name__ == '__main__':
    main()