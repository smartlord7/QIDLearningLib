import matplotlib

import QIDLearningLib.test.qid_specific as qt
import QIDLearningLib.test.causality as ct
import QIDLearningLib.test.data_privacy as dpt
import QIDLearningLib.test.data_utility as dut
import QIDLearningLib.test.performance as pt

def main():
    matplotlib.use('TkAgg')
    qt.main()
    ct.main()
    dpt.main()
    dut.main()
    pt.main()


if __name__ == "__main__":
    main()
