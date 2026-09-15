"""Family-tree primitives (parents, spouse, sex) and the relations derived from them."""

import csv
from dataclasses import dataclass
from pathlib import Path

DATA_PATH = Path(__file__).parents[2] / "data" / "families.csv"

# families.csv has no header row
FIELDS = ("id", "name", "family", "sex", "father", "mother", "spouse")


@dataclass(frozen=True)
class Family:
    parents: dict[str, tuple[str, str]]  # person -> (father, mother)
    spouse: dict[str, str]
    sex: dict[str, str]  # "M" or "F"


def load_families(path: Path = DATA_PATH) -> Family:
    parents = {}
    spouse = {}
    sex = {}
    with open(path, newline="") as csvfile:
        for row in csv.DictReader(csvfile, fieldnames=FIELDS):
            name = row["name"]
            sex[name] = row["sex"]
            if row["father"] and row["mother"]:
                parents[name] = (row["father"], row["mother"])
            if row["spouse"]:
                spouse[name] = row["spouse"]
    return Family(parents, spouse, sex)


# Sex-neutral relations, built directly on the primitives


def parents_of(person: str, family: Family) -> list[str]:
    return list(family.parents.get(person, ()))


def children_of(person: str, family: Family) -> list[str]:
    return [child for child, parents in family.parents.items() if person in parents]


def siblings_of(person: str, family: Family) -> list[str]:
    # anyone sharing at least one parent, so half-siblings count too
    own_parents = set(parents_of(person, family))
    return [
        child
        for child, parents in family.parents.items()
        if child != person and own_parents.intersection(parents)
    ]


def spouse_of(person: str, family: Family) -> list[str]:
    return [family.spouse[person]] if person in family.spouse else []


# Gendered relations: a sex-neutral relation filtered by the sex of the result


def _filter_sex(people: list[str], family: Family, sex: str) -> list[str]:
    return [p for p in people if family.sex[p] == sex]


def father(person: str, family: Family) -> list[str]:
    return _filter_sex(parents_of(person, family), family, "M")


def mother(person: str, family: Family) -> list[str]:
    return _filter_sex(parents_of(person, family), family, "F")


def son(person: str, family: Family) -> list[str]:
    return _filter_sex(children_of(person, family), family, "M")


def daughter(person: str, family: Family) -> list[str]:
    return _filter_sex(children_of(person, family), family, "F")


def brother(person: str, family: Family) -> list[str]:
    return _filter_sex(siblings_of(person, family), family, "M")


def sister(person: str, family: Family) -> list[str]:
    return _filter_sex(siblings_of(person, family), family, "F")


def husband(person: str, family: Family) -> list[str]:
    return _filter_sex(spouse_of(person, family), family, "M")


def wife(person: str, family: Family) -> list[str]:
    return _filter_sex(spouse_of(person, family), family, "F")
