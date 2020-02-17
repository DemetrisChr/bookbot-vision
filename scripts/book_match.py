# Temporary list until a database is set up
# label_codes = ["F\n128\n.65\n.C3\nCoo", "F\n229\nVau.", "F\n334\n.S4\nSwa.", "E\n441\nAsh.", "F\n1025\nNor.", "F\n394\n.H857\nCar."]
label_codes = ['DG311 Gib.', 'BJ1499.S5 Kag.', 'QC21.3 Hal.', 'QC174.12 Bra.', 'PS3562.E353 Lee.', 'PR4662 Eli.', 'HA29 Huf.', 'QA276 Whe.' ,'QA76.73.H37 Lip.', 'QA76.62 Bir.']
# label_codes = [code.replace('\n', '') for code in label_codes]
maxlength = max([len(code) for code in label_codes])
label_codes = [code.ljust(maxlength, ".") for code in label_codes]
# book_names = ["A description of the New York Central Park",
#              "American genesis",
#              "The Selma Campaign",
#              "Slavery, Capitalism, and Politics",
#              "Northern Exposures",
#              "Red Scare"]
book_names = ['The decline and fall of the Roman Empire', 
            'Silence: in the age of noise',
            'Principles of Physics',
            'Quantum Mechanics',
            'To kill a mockingbird',
            'Middlemarch',
            'How to lie with statistics',
            'Naked statistics: stripping the dread from the data',
            'Learn you a Haskell for great good!',
            'Introduction to functional programming using Haskell']
database = list(zip(label_codes, book_names))

# Calculates the minimum edit distance between s1 and s2
def levenshtein(s1, s2):

    insertion_cost = 1 # Insertions cost one
    deletion_cost = 0 # Deleting is free
    substitution_cost = 1 # Substitution same cost as sum

    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + insertion_cost
            deletions = current_row[j] + deletion_cost
            substitutions = previous_row[j] + (c1 != c2) *  substitution_cost
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

# Returns the true label code, name of the book which is closest match to input read_label from the datbase and the cost
def closest_label_match(read_label):
    min_cost = 1000
    min_index = -1
    for i in range(0,len(database)):
        code = database[i][0]
        MED = levenshtein(code,read_label)
        if(MED<min_cost):
            min_cost = MED
            min_index = i
    print(min_cost)
    # Cutoff point 20, if higher cost empty name is returned
    if(min_cost>20):
        return "", 1000
    return database[min_index][0], database[min_index][1], min_cost
