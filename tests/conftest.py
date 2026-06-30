"""pytest configuration: sets the matplotlib backend before any test module imports pyplot.

pytest loads conftest.py before any test module, so this runs early enough
to satisfy matplotlib's "select backend before pyplot import" requirement.

"""

import matplotlib

matplotlib.use("Agg")
