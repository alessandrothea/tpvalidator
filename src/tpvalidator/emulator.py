import numpy as np
import numpy.typing as npt


def div(a, b):
    vb = np.int32((1 << 15) / b)
    mulhrs = np.int32(np.int32(a) * vb)
    mulhrs = (mulhrs >> 14) + 1
    mulhrs = mulhrs >> 1
    return np.int16(mulhrs)


class PedSub:
    def __init__(self, baseline_estimate: np.int16) -> None:
        """
        Parameters:
            baseline_estimate (np.int16):
                An estimate of what the baseline should start at.
        """
        self.pedestal: np.int16 = baseline_estimate
        self.pedestals: list[np.int16] = [self.pedestal]
        self.accum: int = 0
        self._accum_limit: int = 10
        return

    def __call__(self, x: np.int16) -> np.int16:
        """
        Apply the pedestal subtraction and update values.
        """
        if x > self.pedestal:
            self.accum += 1
        elif x < self.pedestal:
            self.accum -= 1

        if self.accum >= self._accum_limit:
            self.pedestal += 1
            self.accum = 0
        elif self.accum <= -self._accum_limit:
            self.pedestal -= 1
            self.accum = 0

        self.pedestals.append(self.pedestal)  # Useful to keep track of.
        return x - self.pedestal


class AbsRS:
    def __init__(self, wf: npt.NDArray[np.uint16], memory_factor: int, scale_factor: int) -> None:
        """
        Parameters:
            wf (NDArray[np.uint16]):
                The waveform to apply the absolute running sum to.
            memory_factor (int):
                The memory factor for the running sum.
            scale_factor (int):
                The scale factor for the signal value.
        """
        self.wf: npt.NDArray[np.int16] = wf.astype(np.int16)
        self._mem_factor: int = memory_factor
        self._scale_factor: int = scale_factor

        self._baseline_estimate: np.int16 = np.median(self.wf).astype(np.int16)
        self._pedsub0: PedSub = PedSub(self._baseline_estimate)
        self._pedsub1: PedSub = PedSub(0)  # Awkward to estimate :(

        self._rs: np.uint16 = wf[0] - self._baseline_estimate
        return

    def process(self) -> npt.NDArray[np.uint16]:
        """
        Process the given waveform according to the known AbsRS definition.
        """
        abs_rs: npt.NDArray[np.int16] = np.zeros(self.wf.shape, dtype=np.int16)
        abs_rs[0] = self._rs
        for idx in range(1, len(self.wf)):
            x: np.int16 = self.wf[idx]
            ps = self._pedsub0(x)
            self._rs = self._mem_factor * div(self._rs, 10) + self._scale_factor * div(np.abs(ps), 10)
            ps = self._pedsub1(self._rs)
            abs_rs[idx] = ps
        return abs_rs, self._pedsub0.pedestals, self._pedsub1.pedestals

