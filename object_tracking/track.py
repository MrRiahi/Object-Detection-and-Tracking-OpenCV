from object_tracking.config import Config as Cfg
from object_tracking.tracker import MILTracker, KCFTracker, CSRTTracker, MOSSETacker, TLDTracker, \
                                    BoostingTracker, DaSiamRPNTacker, GOTURNTracker, DeepSortTracker


def get_tracker():
    """
    This function gets the tracker object and returns it
    :return:
    """

    if Cfg.TRACKER_TYPE == 'MIL':
        tracker = MILTracker()

    elif Cfg.TRACKER_TYPE == 'KCF':
        tracker = KCFTracker()

    elif Cfg.TRACKER_TYPE == 'CSRT':
        tracker = CSRTTracker()

    elif Cfg.TRACKER_TYPE == 'MOSSE':
        tracker = MOSSETacker()

    elif Cfg.TRACKER_TYPE == 'TLD':
        tracker = TLDTracker()

    elif Cfg.TRACKER_TYPE == 'Boosting':
        tracker = BoostingTracker()

    elif Cfg.TRACKER_TYPE == 'DaSiamRPN':
        tracker = DaSiamRPNTacker()

    elif Cfg.TRACKER_TYPE == 'GOTURN':
        tracker = GOTURNTracker()

    else:
        raise Exception('Invalid tracker type!')

    return tracker
