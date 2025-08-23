#!/bin/python3

from hdf5libs import HDF5RawDataFile
from daqdataformats import Fragment
from detchannelmaps import make_tpc_map, TPCChannelMap
from detdataformats import DAQEthHeader
from fddetdataformats import WIBEthFrame
from rawdatautils.unpack.wibeth import np_array_adc

import click
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from pathlib import Path


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


def get_wibeth_paths(fragment_paths: list[str]) -> list[str]:
    """ Get only the WIBEthFrame fragment paths. """
    wibeth_paths: list[str] = []
    for path in fragment_paths:
        if path.endswith("WIBEth"):
            wibeth_paths.append(path)
    return wibeth_paths


def get_tpc_coords(frame: WIBEthFrame) -> tuple[int, int, int, int]:
    """
    Get the TPC coordinates from the WIBEthFrame.

    Parameters:
        frame (WIBEthFrame):
            The WIBEthFrame being processed.

    Returns:
        tuple[int, int, int, int]:
            The (detector ID, crate ID, slot ID, stream ID) for this frame.
    """
    daq_header: DAQEthHeader = frame.get_daqheader()
    return (daq_header.det_id, daq_header.crate_id, daq_header.slot_id, daq_header.stream_id)


def plot(wf: npt.NDArray[np.uint16], absrs: npt.NDArray[np.int16], ps0: npt.NDArray[np.int16], ps1: npt.NDArray[np.int16]) -> None:
    pedsub: PedSub = PedSub(np.median(wf).astype(np.int16))
    y: npt.NDArray[np.int16] = np.zeros(wf.shape, dtype=np.int16)
    pedestals: npt.NDArray[np.int16] = np.zeros(wf.shape, dtype=np.int16)
    for idx in range(len(wf)):
        pedestals[idx] = pedsub.pedestal
        y[idx] = pedsub(wf[idx])

    # WF & AbsRS plot.
    plt.figure(figsize=(6, 4), dpi=300, layout="constrained")
    plt.box(False)

    plt.plot(y, '-hk', ms=2, label=f"WF RMS = {np.std(wf):.2f}")
    plt.plot(absrs, '-h', c="#EE442F", ms=2, label=f"AbsRS RMS = {np.std(absrs):.2f}")

    plt.title("Python AbsRS & WF")
    plt.xlabel("Readout Ticks (512 ns / tick)")
    plt.ylabel("Signal Amplitude (ADC)")

    plt.legend()

    plt.savefig("wf_absrs.png")
    plt.close()

    # Pedestal plot.
    plt.figure(figsize=(6, 4), dpi=300, layout="constrained")
    plt.box(False)

    plt.plot(ps0, '-h', c="#EE442F", ms=2, label="PedSub 0")
    plt.plot(ps1, '-h', c="#63ACBE", ms=2, label="PedSub 1")

    plt.title("AbsRS & WF Pedestals")
    plt.xlabel("Readout Ticks (512 ns / tick)")
    plt.ylabel("Signal Pedestal (ADC)")

    plt.legend()

    plt.savefig("pedestals.png")
    plt.close()
    exit()
    return


@click.command()
@click.argument("file_path", type=click.Path(readable=True, path_type=Path))
@click.option("--channel-map", '-c', "channel_map_name", type=str, default="PD2VDTPCChannelMap", help="Detector channel map to use.")
@click.option("--memory-factor", "-m", type=int, default=8)
@click.option("--scale-factor", '-s', type=int, default=5)
def main(file_path: Path, channel_map_name: str, memory_factor: int, scale_factor: int) -> None:
    # Preliminary objects.
    file: HDF5RawDataFile = HDF5RawDataFile(str(file_path.expanduser()))
    channel_map: TPCChannelMap = make_tpc_map(channel_map_name)

    wibeth_paths: list[str] = get_wibeth_paths(file.get_all_fragment_dataset_paths())
    for path in tqdm(wibeth_paths, total=len(wibeth_paths)):
        # Reading and staging.
        frag: Fragment = file.get_frag(path)
        tpc_coords: tuple[int, int, int, int] = get_tpc_coords(WIBEthFrame(frag.get_data()))
        channels: npt.NDArray[int] = np.array([channel_map.get_offline_channel_from_det_crate_slot_stream_chan(*tpc_coords, chan) for chan in range(64)], dtype=int)
        adcs: npt.NDArray[np.uint16] = np_array_adc(frag).astype(np.uint16).T

        for wf, channel in zip(adcs, channels):
            if channel % 3072 > 972 * 2:  # Intentionally getting induction.
                continue
            # Get something interesting.
            if np.max(wf.astype(np.int16) - np.median(wf).astype(np.int16)) > 3000:
                absrs_runner: AbsRS = AbsRS(wf, memory_factor, scale_factor)
                absrs, ps0, ps1 = absrs_runner.process()
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!
                plot(wf, absrs, ps0, ps1)   # <- EXITS. Only cared for one plot as an example.
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!
    return


if __name__ == "__main__":
    main()
