import sys
import sacrebleu

def bleu_sentence(sys, ref, order=4):
    if order !=4:
        raise NotImplementedError
    return sacrebleu.sentence_bleu(sys, [ref]).score

def main(argv):
    # Sentence Level BLEU
    with open(argv[0], 'r', encoding='utf-8') as label, \
            open(argv[1], 'r', encoding='utf-8') as src, \
            open(argv[2], 'r', encoding='utf-8') as ref, \
            open(argv[3], 'r', encoding='utf-8') as t1, \
            open(argv[4], 'r', encoding='utf-8') as t2, \
            open(argv[5], 'r', encoding='utf-8') as t3, \
            open(argv[6], 'r', encoding='utf-8') as t4, \
            open(argv[7], 'r', encoding='utf-8') as t5, \
            open(argv[8], 'r', encoding='utf-8') as t6, \
            open(argv[9], 'r', encoding='utf-8') as t7, \
            open(argv[10], 'r', encoding='utf-8') as t8, \
            open(argv[11], 'r', encoding='utf-8') as t9, \
            open(argv[12], 'r', encoding='utf-8') as tx:
        for idx, (lab,s,r,l1,l2,l3,l4,l5,l6,l7,l8,l9,lx) in enumerate(zip(label,src,ref,t1,t2,t3,t4,t5,t6,t7,t8,t9,tx)):
            bleu1 = bleu_sentence(l1.lower(), r.lower())
            bleu2 = bleu_sentence(l2.lower(), r.lower())
            bleu3 = bleu_sentence(l3.lower(), r.lower())
            bleu4 = bleu_sentence(l4.lower(), r.lower())
            bleu5 = bleu_sentence(l5.lower(), r.lower())
            bleu6 = bleu_sentence(l6.lower(), r.lower())
            bleu7 = bleu_sentence(l7.lower(), r.lower())
            bleu8 = bleu_sentence(l8.lower(), r.lower())
            bleu9 = bleu_sentence(l9.lower(), r.lower())
            bleu10 = bleu_sentence(lx.lower(), r.lower())

            bleu_list = [bleu1, bleu2, bleu3, bleu4, bleu5, bleu6, bleu7, bleu8, bleu9]

            # if all(i < bleu10 for i in bleu_list): #and bleu10 >=100:
            #     if all(i < bleu9 for i in bleu_list[:-1]): #and bleu10 >= 100:
            #         print(f'{bleu10} {bleu_list} {idx}')
            #         print(f'{lab} {s} {r}')
            #         print(f'{l1} {l2} {l3} {l4} {l5} {l6} {l7} {l8} {l9} {lx}')


            if all(i < bleu10 for i in bleu_list) and bleu10 >=100:
                print(f'{bleu10} {bleu_list} {idx}')
                print(f'{lab} {s} {r}')
                print(f'{l1} {l2} {l3} {l4} {l5} {l6} {l7} {l8} {l9} {lx}')


if "__main__" == __name__:
    main(sys.argv[1:])