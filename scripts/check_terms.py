from translation.dictionary import legal_dict_kn_en

terms = ['ಕೋರ್ಟ್', 'ನೀಡಿತು', 'ಪಕ್ಷಗಳು', 'ಒಪ್ಪಂದ', 'ನಿಯಮಗಳು', 'ಪಾಲಿಸುವಿಕೆ']

for t in terms:
    trans = legal_dict_kn_en.get(t, "NOT FOUND")
    print(f'{t}: {trans}')
