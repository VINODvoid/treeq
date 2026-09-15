import pytest

from treeq.kinship import (
    brother,
    children_of,
    daughter,
    father,
    husband,
    load_families,
    mother,
    parents_of,
    siblings_of,
    sister,
    son,
    spouse_of,
    wife,
)


@pytest.fixture(scope="module")
def family():
    return load_families()


def test_primitive_counts(family):
    assert len(family.parents) == 16
    assert len(family.spouse) == 12
    assert len(family.sex) == 24


@pytest.mark.parametrize(
    "relation, person, expected",
    [
        # English family
        (father, "David", {"John"}),
        (father, "Sarah", {"John"}),
        (father, "James", {"David"}),
        (father, "Alice", {"Michael"}),
        (mother, "David", {"Mary"}),
        (mother, "Sarah", {"Mary"}),
        (mother, "James", {"Linda"}),
        (mother, "Alice", {"Sarah"}),
        (son, "John", {"David"}),
        (son, "David", {"James", "Robert"}),
        (daughter, "John", {"Sarah"}),
        (daughter, "David", {"Emma"}),
        (brother, "James", {"Robert"}),
        (sister, "James", {"Emma"}),
        (husband, "Mary", {"John"}),
        (husband, "David", set()),
        (wife, "John", {"Mary"}),
        (wife, "David", {"Linda"}),
        # Italian family
        (father, "Marco", {"Giovanni"}),
        (mother, "Marco", {"Maria"}),
        (son, "Giovanni", {"Marco"}),
        (daughter, "Giovanni", {"Giulia"}),
        (brother, "Matteo", {"Andrea"}),
        (sister, "Matteo", {"Sofia"}),
        # people at the edges of the tree
        (father, "John", set()),
        (son, "James", set()),
        (brother, "Linda", set()),
        (wife, "Emma", set()),
    ],
)
def test_relation(family, relation, person, expected):
    result = relation(person, family)
    assert len(result) == len(set(result)), "duplicates in result"
    assert set(result) == expected


def test_relations_are_consistent_inverses(family):
    for person in family.sex:
        for parent in parents_of(person, family):
            assert person in children_of(parent, family)
        for partner in spouse_of(person, family):
            assert spouse_of(partner, family) == [person]
        for sibling in siblings_of(person, family):
            assert person in siblings_of(sibling, family)
