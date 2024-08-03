import matplotlib

import test.qid_specific as qt
import test.causality as ct
import test.data_privacy as dpt
import test.data_utility as dut
import test.performance as pt

def main():
    matplotlib.use('TkAgg')
    dpt.main()
    #qt.main()
    #ct.main()
    #dut.main()
    #pt.main()


if __name__ == "__main__":
    main()
