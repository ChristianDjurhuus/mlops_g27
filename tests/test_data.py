from datasets import load_from_disk

class TestClass:
    data = load_from_disk("data/processed")
    

    # GET DATA FOR FIT STAGE
    full_train_dataset = data["train"]
    full_valid_dataset = data["valid"]
    full_test_dataset = data["test"]

    full_train_dataset.num_rows

    # Supposed number of data
    N_train = 25000
    N_test = 12500
    N_valid = 12500

    # Testing trainingdata
    def test_traindata(self):
        # Number of documents in train corpus
        labels = self.full_train_dataset["label"]
        assert (
            self.full_train_dataset.num_rows == self.N_train
        ), "Train data did not have the correct number of documents"

        # Labels
        assert (
            len(labels) == self.N_train
        ), "Train data did not have the correct number of labels"
        assert all(
            i in set(labels) for i in range(1)
        ), "At least one train data label wasn't correct."

    def test_valid_data(self):
        # Image structure
        labels = self.full_valid_dataset["label"]
        assert self.full_valid_dataset.num_rows == self.N_valid, (
            f"Validation data did not have the correct number of documents, "
            f"but had: {len(self.full_valid_dataset)}"
        )

        # Labels
        assert (
            len(labels) == self.N_valid
        ), "Test data did not have the correct number of labels"
        assert all(
            i in set(labels) for i in range(1)
        ), "At least one validation data label wasn't correct."

    def test_testdata(self):
        # Image structure
        labels = self.full_test_dataset["label"]
        assert self.full_test_dataset.num_rows == self.N_test, (
            f"Test data did not have the correct number of documents, "
            f"but had: {len(self.full_test_dataset)}"
        )

        # Labels
        assert (
            len(labels) == self.N_test
        ), "Test data did not have the correct number of labels"
        assert all(
            i in set(labels) for i in range(1)
        ), "At least one test data label wasn't correct."
