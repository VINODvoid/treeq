import numpy as np
import pytest

from treeq import kinship
from treeq.dataset import RELATION_NAMES, build_dataset

N_PEOPLE = 24
N_RELATIONS = 8
N_ROWS = 92  # (person, relation) pairs that have at least one answer
N_ANSWERS = 104  # total correct answers across those rows


@pytest.fixture(scope="module")
def family():
    return kinship.load_families()


@pytest.fixture(scope="module")
def dataset(family):
    return build_dataset(family)


def test_relation_names_are_the_eight_implemented_relations():
    assert RELATION_NAMES == (
        "father",
        "mother",
        "husband",
        "wife",
        "son",
        "daughter",
        "brother",
        "sister",
    )


def test_shapes(dataset):
    assert dataset.person_in.shape == (N_ROWS, N_PEOPLE)
    assert dataset.relation_in.shape == (N_ROWS, N_RELATIONS)
    assert dataset.target.shape == (N_ROWS, N_PEOPLE)
    assert len(dataset.pairs) == N_ROWS


def test_arrays_are_float(dataset):
    # the network multiplies these, so they must not be integer or object arrays
    for array in (dataset.person_in, dataset.relation_in, dataset.target):
        assert array.dtype.kind == "f"


def test_arrays_hold_only_zeros_and_ones(dataset):
    for array in (dataset.person_in, dataset.relation_in, dataset.target):
        assert np.isin(array, (0.0, 1.0)).all()


def test_inputs_are_one_hot(dataset):
    assert (dataset.person_in.sum(axis=1) == 1).all()
    assert (dataset.relation_in.sum(axis=1) == 1).all()


def test_targets_are_multi_hot_and_never_empty(dataset):
    assert (dataset.target.sum(axis=1) >= 1).all()
    assert dataset.target.sum() == N_ANSWERS


def test_people_are_the_csv_order(dataset, family):
    assert dataset.people == tuple(family.sex)
    assert len(set(dataset.people)) == N_PEOPLE


def test_relations_match_the_module_constant(dataset):
    assert dataset.relations == RELATION_NAMES


def test_index_maps_round_trip(dataset):
    for index, name in enumerate(dataset.people):
        assert dataset.person_index[name] == index
    for index, name in enumerate(dataset.relations):
        assert dataset.relation_index[name] == index


def test_pairs_are_unique(dataset):
    assert len(set(dataset.pairs)) == len(dataset.pairs)


def test_david_has_two_sons_in_one_row(dataset):
    row = dataset.pairs.index(("David", "son"))

    assert dataset.person_in[row].argmax() == dataset.person_index["David"]
    assert dataset.relation_in[row].argmax() == dataset.relation_index["son"]

    answers = {dataset.people[i] for i in np.flatnonzero(dataset.target[row])}
    assert answers == {"James", "Robert"}


def test_every_row_agrees_with_kinship(dataset, family):
    for row, (person, relation) in enumerate(dataset.pairs):
        expected = set(getattr(kinship, relation)(person, family))

        assert expected, f"row {row} has no answer and should not be in the dataset"
        assert dataset.people[dataset.person_in[row].argmax()] == person
        assert dataset.relations[dataset.relation_in[row].argmax()] == relation

        encoded = {dataset.people[i] for i in np.flatnonzero(dataset.target[row])}
        assert encoded == expected, f"row {row}: {person} {relation}"


def test_dataset_is_reproducible(family):
    first, second = build_dataset(family), build_dataset(family)

    assert first.people == second.people
    assert first.relations == second.relations
    assert first.pairs == second.pairs
    for name in ("person_in", "relation_in", "target"):
        assert np.array_equal(getattr(first, name), getattr(second, name))
