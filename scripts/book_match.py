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

    # Returns the name of the book which is closest match to input read_label from the datbase
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
        return database[min_index][1], min_cost
