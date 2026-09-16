import numpy as np
from dataclasses import dataclass
from  treeq.kinship import Family, load_families,son
from treeq import kinship
RELATION_NAMES = ("father", "mother", "husband", "wife", "son", "daughter", "brother", "sister")



@dataclass(frozen=True)
class Dataset:
    people : tuple[str,...]
    relations : tuple[str,...]
    person_index:dict[str,int]
    relation_index : dict[str,int]
    person_in : np.ndarray
    relation_in :np.ndarray
    target: np.ndarray
    pairs:tuple[tuple[str,str],...]


def build_dataset(family:Family) -> Dataset:
    rows = []
    people = tuple(family.sex)
    for person in people:
        for relation in RELATION_NAMES:
            relation_fn = getattr(kinship,relation)
            result = relation_fn(person,family)
            if not result:
                continue
            rows.append((person,relation,result))
    person_index = {person:index for index,person in enumerate(people)}
    relation_index = {relation:index for index,relation in enumerate(RELATION_NAMES)}    
    person_in = np.zeros((len(rows),len(people)))
    relation_in = np.zeros((len(rows), len(RELATION_NAMES)))
    target = np.zeros((len(rows),len(people))) # 92 non-empty relationships X 24 different persons 
    for row, (person, relation, answers) in enumerate(rows):
        person_in[row, person_index[person]] = 1.0
        relation_in[row,relation_index[relation]] = 1.0 
        for name in answers:
            target[row,person_index[name]] = 1.0
    pairs = tuple((person,relation) for person,relation,_ in rows)
    return Dataset(
            people=people,
            relations=RELATION_NAMES,
            person_index=person_index,
            relation_index=relation_index,
            person_in=person_in,
            relation_in=relation_in,
            target=target,
            pairs=pairs)

