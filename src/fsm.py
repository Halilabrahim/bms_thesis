from enum import Enum
from typing import Dict


class BMSState(str, Enum):
    OFF = "OFF"
    RUN = "RUN"
    FAULT = "FAULT"
    EMERGENCY_SHUTDOWN = "EMERGENCY_SHUTDOWN"


class BMSStateMachine:
    """
    Very simple BMS state machine.

    Transitions:
      - OFF  --(enable)--> RUN
      - RUN  --(any_fault)--> FAULT
      - RUN  --(emergency)--> EMERGENCY_SHUTDOWN
      - FAULT --(emergency)--> EMERGENCY_SHUTDOWN
      - EMERGENCY_SHUTDOWN: latched until power cycle (no exit).
    """

    def __init__(self):
        self.state = BMSState.RUN  # start in RUN for our simulations

    def reset(self, initial: BMSState = BMSState.RUN) -> None:
        self.state = initial

    def step(self, fault_flags: Dict[str, bool], enable: bool = True) -> BMSState:
        if not enable:
            self.state = BMSState.OFF
            return self.state

        # OFF -> RUN (power on)
        if self.state == BMSState.OFF and enable:
            self.state = BMSState.RUN

        # RUN transitions
        if self.state == BMSState.RUN:
            if fault_flags.get("emergency", False):
                self.state = BMSState.EMERGENCY_SHUTDOWN
            elif fault_flags.get("any_fault", False):
                self.state = BMSState.FAULT

        # FAULT transitions
        elif self.state == BMSState.FAULT:
            if fault_flags.get("emergency", False):
                self.state = BMSState.EMERGENCY_SHUTDOWN

        # EMERGENCY_SHUTDOWN: latched, no exit

        return self.state
