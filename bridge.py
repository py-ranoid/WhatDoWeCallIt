import numpy as np

MINLEFT = 3
MINRIGHT = 3
SEQLEN = 3
COMPARE = 3
LEFT_BIAS = [0.07, 0.04, 0.02]


def one_hot(word):
    x_pred = np.zeros((1, SEQLEN, len(chars)))

    for t, char in enumerate(word):
        x_pred[0, t, char_indices[char]] = 1.
    return x_pred


def sample_preds(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # probas = np.random.multinomial(1, preds, 1)
    # return np.argmax(probas)
    return preds


def proc(left, right, verbose=False):
    best_matches = {}
    best_i = None
    best_i_score = -1
    for i in range(0, len(left) - MINLEFT + 1):
        # print ("I:", i)
        # Searching all sequences of size COMPARE in the right word
        # to find best match
        best_j = None
        best_j_score = -1
        for j in range(0, len(right) - MINRIGHT + 1):
            right_chars = right[j:j + COMPARE]
            s = 0
            for x in range(COMPARE):
                # Character on right which is being sampled
                c_index = char_indices[right_chars[x]]
                if verbose:
                    print ("Sampling " + left[i + x:i + SEQLEN] +
                           right[j:j + x] + "-->" + right_chars[x])

                # Generating sequence and getting probability
                Xoh = one_hot(left[i + x:i + SEQLEN] + right[j:j + x])
                preds = model.predict(Xoh, verbose=0)[0]
                pred_probs = sample_preds(preds, 0.7)

                # Getting corresponding character in left word
                left_char = np.zeros((1, len(chars)))
                try:
                    left_char[0, char_indices[left[i + SEQLEN + x]]] = 1
                except IndexError:
                    pass
                # Adding some bias to left_char and adding it to predicted probs
                biased_probs = LEFT_BIAS[x] * left_char + \
                    (1 - LEFT_BIAS[x]) * pred_probs
                # l_preds.append(biased_probs)

                # Adding probability of bridging at c_index to s
                s += biased_probs[0, c_index]
            if verbose:
                print (i, j, s,)
            if s > best_j_score:
                best_j = j
                best_j_score = s
        best_matches[i] = {'index': best_j, 'score': best_j_score}
        if best_j_score > best_i_score:
            best_i_score = best_j_score
            best_i = i

    return best_matches, best_i


def bridge(left, right, verbose=False):
    matches, i = proc(left, right, verbose)
    print ("Best (" + str(round(matches[i]['score'], 4)) + "): " +
           (left[:i + SEQLEN] + right[matches[i]['index']:]))
    print ("Others :")
    for i_temp in matches:
        print ("(" + str(round(matches[i_temp]['score'], 4)) + "): " +
               left[:i_temp + SEQLEN] + right[matches[i_temp]['index']:])
    # print (matches)
