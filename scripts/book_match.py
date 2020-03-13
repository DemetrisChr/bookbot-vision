label_codes_original = ['DG311 Gib.', 'BJ1499.S5 Kag.', 'QC21.3 Hal.', 'QC174.12 Bra.', 'PS3562.E353 Lee.',
                        'PR4662 Eli.', 'HA29 Huf.', 'QA276 Whe.', 'QA76.73.H37 Lip.', 'QA76.62 Bir.']


class BookMatch():
    def __init__(self, all_labels=label_codes_original, insertion_cost=1, deletion_cost=0, substitution_cost=1):
        self.all_labels_original = all_labels
        maxlength = max([len(code) for code in self.all_labels_original])
        self.all_labels = all_labels
        self.all_labels = [code.ljust(maxlength, ".") for code in self.all_labels]
        self.insertion_cost = insertion_cost
        self.deletion_cost = deletion_cost
        self.substitution_cost = substitution_cost

    def levenshtein(self, s1, s2):
        """
        Calculates the levenshtein distance between the two strings
        """
        if len(s1) < len(s2):
            return self.levenshtein(s2, s1)

        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + self.insertion_cost
                deletions = current_row[j] + self.deletion_cost
                substitutions = previous_row[j] + (c1 != c2) * self.substitution_cost
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def closest_label_match(self, read_label):
        """
        Returns the true label code, name of the book which is closest match to input read_label from the datbase and the cost
        """
        min_cost = 1000
        min_index = -1
        for i in range(len(self.all_labels)):
            code = self.all_labels[i]
            MED = self.levenshtein(code, read_label)
            if(MED < min_cost):
                min_cost = MED
                min_index = i

        # Cutoff point 20, if higher cost empty name is returned
        if(min_cost > 15):
            return "", min_cost
        return self.all_labels_original[min_index], min_cost
