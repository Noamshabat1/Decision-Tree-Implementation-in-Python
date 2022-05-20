#################################################################
# FILE : snake.py
# WRITER 1 : noam shabat , no.amshabat1 , 206515579
# EXERCISE : intro2cs2 Noam 2021
# DESCRIPTION: A file that creates s decision tree.
# STUDENTS I DISCUSSED THE EXERCISE WITH: ...
# WEB PAGES I USED: stackoverflow.com
#################################################################
import itertools


class Node:
    def __init__(self, data, positive_child=None, negative_child=None):
        """

        :param data: the data that cell holds.
        :param positive_child: the right closest node to the one that the
        pointer is on.
        :param negative_child:the left closest node to the one that the
        pointer is on.
        """
        self.data = data
        self.positive_child = positive_child
        self.negative_child = negative_child


class Record:
    def __init__(self, illness, symptoms):
        """

        :param illness: a string object that contains an illness name
        :param symptoms: a list of strings objects that contains symptom's
        name's.
        """
        self.illness = illness
        self.symptoms = symptoms


def parse_data(filepath):
    """

    :param filepath:
    :return:
    """
    with open(filepath) as data_file:
        records = []
        for line in data_file:
            words = line.strip().split()
            records.append(Record(words[0], words[1:]))
        return records


def node_equality(node_a, node_b):
    if node_a.data != node_b.data:
        return False
    diag_a = Diagnoser(node_a)
    diag_b = Diagnoser(node_b)
    if diag_a.all_illnesses() != diag_b.all_illnesses():
        return False
    for illness in diag_a.all_illnesses():
        path_to_a = diag_a.paths_to_illness(illness)
        path_to_b = diag_b.paths_to_illness(illness)
        if path_to_a != path_to_b:
            return False
    return True


class Diagnoser:
    def __init__(self, root: Node):
        """
        this class is creating a Diagnostic for the program
        :param root: the root of tree that we are examining.
        """
        self.root = root

    def diagnose(self, symptoms):
        """
        this method is doing the diagnostic op on the tree.
        :param symptoms: a list of symptoms.
        :return: the illness of the patient
        """
        current = self.root
        symptoms_set = set(symptoms)
        current_next = current.positive_child
        while current_next is not None:
            if current.data in symptoms_set:
                current = current.positive_child
                current_next = current.positive_child
            elif current.data not in symptoms_set:
                current = current.negative_child
                current_next = current.positive_child
        return current.data

    def calculate_success_rate(self, records):
        """
        this method is calculating the success rate of the func diagnose.
        :param records: a list of records that contains symptoms list and an
        illness.
        :return: the ration.
        """
        len_of_records = len(records)
        counter = 0
        if len_of_records == 0:
            raise ValueError("The record list is empty!")
        for rec in records:
            if self.diagnose(rec.symptoms) == rec.illness:
                counter += 1
            else:
                continue
        return counter / len_of_records

    def all_illnesses(self):
        """
        this method is returning all the illnesses that there is.
        :return: a list sorted by the frequency of there appearance.
        """
        illnesses_dic = dict()

        def recursive_engine(node: Node):
            if node.positive_child is None:
                if node.data is None:
                    return
                if node.data in illnesses_dic:
                    illnesses_dic[node.data] += 1
                else:
                    illnesses_dic[node.data] = 1
                return
            recursive_engine(node.positive_child)
            recursive_engine(node.negative_child)

        recursive_engine(self.root)

        illness_ls = [(k, v) for k, v in illnesses_dic.items()]
        illness_ls.sort(key=lambda x: x[1], reverse=True)
        sorted_illnesses_dic = [k for k, v in illness_ls]

        return sorted_illnesses_dic

    def paths_to_illness(self, illness):
        """
        this func is list that contains the path to the illness.
        :param illness: a string description of an illness
        :return: a list.
        """
        total_il = []
        self.helper_paths_to_illness(illness, self.root, total_il, [])
        return total_il

    def helper_paths_to_illness(self, illness, root, total_il, small_ls):
        current = root
        if current.positive_child is None:
            if current.data is not illness:
                return

            total_il.append(small_ls)
            return

        self.helper_paths_to_illness(illness, root.positive_child, total_il,
                                     small_ls + [True])
        self.helper_paths_to_illness(illness, root.negative_child, total_il,
                                     small_ls + [False])
        return

    def minimize(self, remove_empty=False):
        """
        :param remove_empty:
        :return:
        """

        def recursive_engine(node: Node):
            if node.positive_child is None:
                return node
            node.positive_child = recursive_engine(node.positive_child)
            node.negative_child = recursive_engine(node.negative_child)
            if remove_empty:
                if not Diagnoser(node.positive_child).all_illnesses():
                    if node.negative_child.positive_child is not None:
                        return node.negative_child
                if not Diagnoser(node.negative_child).all_illnesses():
                    if node.positive_child.positive_child is not None:
                        return node.positive_child

            if node_equality(node.positive_child, node.negative_child):
                return node.positive_child
            return node
        self.root = recursive_engine(self.root)


def build_tree(records, symptoms):
    """
    this func is creating a tree.
    :param records: a list of records that contains symptoms list and an
    illness.
    :param symptoms: a list of symptoms.
    :return: a Diagnoser tree object
    """
    set_sy = set(symptoms)
    if len(set_sy) != len(symptoms):
        raise ValueError("incorrect value!")
    for record in records:
        if type(record) != Record:
            raise TypeError("incorrect value!")
    for sy in symptoms:
        if type(sy) != str:
            raise TypeError("incorrect value!")

    def recursive_engine(path, i=0):
        if i >= len(symptoms):
            existing_sy, non_existing_sy = get_existing_non_existing(path)
            valid_records = get_valid_records(existing_sy, non_existing_sy)

            max_illness = get_max_illness(valid_records)

            return Node(max_illness)
        node = Node(symptoms[i])
        node.positive_child = recursive_engine(path + [True], i + 1)
        node.negative_child = recursive_engine(path + [False], i + 1)
        return node

    def get_existing_non_existing(path):
        """
        a help func for tree.
        :param path: .
        :return: .
        """
        existing_sy = set()
        non_existing_sy = set()
        for j in range(len(symptoms)):
            if path[j] is True:
                existing_sy.add(symptoms[j])
            else:
                non_existing_sy.add(symptoms[j])
        return existing_sy, non_existing_sy

    def get_valid_records(existing_sy, non_existing_sy):
        """
        a help func for tree.
        :param existing_sy: .
        :param non_existing_sy: .
        :return: .
        """
        valid_records = []
        for record in records:
            record_valid = True
            set_sy_of_record = set(record.symptoms)
            for sym in existing_sy:
                if sym not in set_sy_of_record:
                    record_valid = False
            for sym in non_existing_sy:
                if sym in set_sy_of_record:
                    record_valid = False
            if record_valid:
                valid_records.append(record)
        return valid_records

    def get_max_illness(valid_records):
        """
        a help func for tree.
        :param valid_records: .
        :return: .
        """
        illnesses = dict()
        for record in valid_records:
            if record.illness in illnesses:
                illnesses[record.illness] += 1
            else:
                illnesses[record.illness] = 1
        max_illness = None
        for ill in illnesses:
            if max_illness is None:
                max_illness = ill
                continue
            if illnesses[ill] > illnesses[max_illness]:
                max_illness = ill
        return max_illness

    return Diagnoser(recursive_engine([]))


def optimal_tree(records, symptoms, depth):
    """
    this func is returning the tree with the best outcome and chance to get
    the correct illness.
    :param records: the record of the illness.
    :param symptoms: a list of symptoms about an illness.
    :param depth: the depth of the leef in the tree.
    :return: this func is returning a Diagnoser tree object.
    """
    set_sy = set(symptoms)
    if depth < 0 or depth > len(symptoms):
        raise ValueError("incorrect value!")
    if len(set_sy) != len(symptoms):
        raise ValueError("incorrect value!")
    for record in records:
        if type(record) != Record:
            raise TypeError("incorrect value!")
    for sy in symptoms:
        if type(sy) != str:
            raise TypeError("incorrect value!")

    dict_max = dict()
    for comb in itertools.combinations(symptoms, depth):
        optimal_tree_object = build_tree(records, list(comb))
        Success_rate = optimal_tree_object.calculate_success_rate(records)
        dict_max[optimal_tree_object] = Success_rate
    best_tree = None
    for tree in dict_max:
        if best_tree is None:
            best_tree = (tree, dict_max[tree])
            continue
        if dict_max[tree] > best_tree[1]:
            best_tree = (tree, dict_max[tree])

    return best_tree[0]


if __name__ == "__main__":
    # Manually build a simple tree.
    #                cough
    #          Yes /       \ No
    #        fever           fever
    #   Yes /     \ No  Yes /    \ No
    # covid-19   cold    cold   covid-19

    # Manually build a simple tree.
    #                cough
    #          Yes /       \ No
    #        None           fever
    #                  Yes /    \ No
    #                 covid-19  healthy

    # Manually build a simple tree.
    #       fever
    #   Yes /    \ No
    # covid-19  healthy
