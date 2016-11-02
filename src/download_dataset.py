from subprocess import call
import os
from urllib.request import urlretrieve

dataset_url = "http://www2.informatik.uni-freiburg.de/~eitel/" \
              "freiburg_groceries_dataset/freiburg_groceries_dataset.tar.gz"

if __name__ == "__main__":
    print("Downloading dataset.")
    urlretrieve(dataset_url, "../freiburg_groceries_dataset.tar.gz")
    print("Extracting dataset.")
    call(["tar", "-xf", "../freiburg_groceries_dataset.tar.gz", "-C", "../"])
    os.remove("../freiburg_groceries_dataset.tar.gz")
    print("Done.")
