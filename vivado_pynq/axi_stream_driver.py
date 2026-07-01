"""AXI-Stream driver for neural network overlay on PYNQ-Z2."""

from datetime import datetime
from typing import Callable, Optional, Tuple, Union

import numpy as np
from pynq import Overlay, allocate

try:
    from tqdm import tqdm
except ImportError:
    # Graceful fallback for environments without tqdm (e.g. bare PYNQ boards).
    def tqdm(iterable, *, desc=None, **kwargs):
        return iterable


class NeuralNetworkOverlay(Overlay):
    """Neural network overlay with single-sample and batch AXI-Stream inference."""

    def __init__(
        self,
        bitfile_name: str,
        x_shape: Tuple[int, ...],
        y_shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        input_dtype: Optional[np.dtype] = None,
        output_dtype: Optional[np.dtype] = None,
        dtbo: Optional[str] = None,
        download: bool = True,
        ignore_version: bool = False,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(
            bitfile_name,
            dtbo=dtbo,
            download=download,
            ignore_version=ignore_version,
            device=device,
        )
        self.sendchannel = self.hier_0.axi_dma_0.sendchannel
        self.recvchannel = self.hier_0.axi_dma_0.recvchannel
        in_dt = input_dtype if input_dtype is not None else dtype
        out_dt = output_dtype if output_dtype is not None else dtype
        self.input_buffer = allocate(shape=x_shape, dtype=in_dt)
        self.output_buffer = allocate(shape=y_shape, dtype=out_dt)

    def _print_dt(
        self, timea: datetime, timeb: datetime, n_samples: int
    ) -> Tuple[float, float]:
        """Print and return inference throughput."""
        dt = timeb - timea
        dts = dt.seconds + dt.microseconds * 10**-6
        rate = n_samples / dts if dts > 0 else float("inf")
        print(
            f"Classified {n_samples} samples in {dts:.6f} seconds "
            f"({rate:.2f} inferences / s)"
        )
        return dts, rate

    def _run_dma_inference(
        self, encoded_sample: np.ndarray, debug: bool = False
    ) -> np.ndarray:
        """Run a single DMA transfer and return the raw output."""
        self.input_buffer[:] = encoded_sample
        self.sendchannel.transfer(self.input_buffer)
        self.recvchannel.transfer(self.output_buffer)
        if debug:
            print("Transfer OK")
        self.sendchannel.wait()
        if debug:
            print("Send OK")
        self.recvchannel.wait()
        if debug:
            print("Receive OK")
        return self.output_buffer.copy()

    def predict(
        self,
        X: np.ndarray,
        debug: bool = False,
        profile: bool = False,
        encode: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        decode: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]:
        """Run hardware inference for one or more samples.

        Args:
            X: Input array. A 1D array processes a single sample with shape
                matching ``x_shape``. A 2D array processes a batch with shape
                ``(batch_size,) + x_shape``.
            debug: Print DMA transfer debug messages.
            profile: Return timing information.
            encode: Function to encode floating-point inputs to the raw AXI-Stream
                data type. Should be vectorized so it works on both 1D and 2D
                arrays.
            decode: Function to decode raw AXI-Stream outputs back to
                floating-point. Should be vectorized so it works on both 1D and
                2D arrays.

        Returns:
            Output array of shape ``y_shape`` for a single sample, or
            ``(batch_size,) + y_shape`` for a batch. If ``profile`` is True,
            returns ``(result, elapsed_seconds, inferences_per_second)``.
        """
        X = np.asarray(X)

        if X.ndim == 1:
            return self._predict_single(X, debug, profile, encode, decode)
        if X.ndim == 2:
            return self._predict_batch(X, debug, profile, encode, decode)

        raise ValueError(
            f"Input must be 1D (single sample) or 2D (batch), got {X.ndim}D"
        )

    def _predict_single(
        self,
        X: np.ndarray,
        debug: bool,
        profile: bool,
        encode: Optional[Callable[[np.ndarray], np.ndarray]],
        decode: Optional[Callable[[np.ndarray], np.ndarray]],
    ) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]:
        """Run inference for a single sample."""
        if profile:
            timea = datetime.now()
        if encode is not None:
            X = encode(X)
        result = self._run_dma_inference(X, debug)
        if decode is not None:
            result = decode(result)
        if profile:
            timeb = datetime.now()
            dts, rate = self._print_dt(timea, timeb, 1)
            return result, dts, rate
        return result

    def _predict_batch(
        self,
        X: np.ndarray,
        debug: bool,
        profile: bool,
        encode: Optional[Callable[[np.ndarray], np.ndarray]],
        decode: Optional[Callable[[np.ndarray], np.ndarray]],
    ) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]:
        """Run inference for a batch of samples with progress bar."""
        expected_shape = self.input_buffer.shape
        if X.shape[1:] != expected_shape:
            raise ValueError(
                f"Input sample shape {X.shape[1:]} does not match expected "
                f"{expected_shape}"
            )

        batch_size = X.shape[0]
        if profile:
            timea = datetime.now()
        if encode is not None:
            X = encode(X)

        iterator = range(batch_size)
        if batch_size > 1:
            iterator = tqdm(range(batch_size), desc="HW inference")

        results = [self._run_dma_inference(X[i], debug) for i in iterator]
        result = np.stack(results)

        if decode is not None:
            result = decode(result)

        if profile:
            timeb = datetime.now()
            dts, rate = self._print_dt(timea, timeb, batch_size)
            return result, dts, rate
        return result
