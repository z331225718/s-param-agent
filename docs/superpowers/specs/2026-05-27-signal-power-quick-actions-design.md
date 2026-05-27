# Signal and Power Quick Actions Design

## Goal

Make the dashboard choose RF quick-action buttons from the loaded Touchstone file instead of always showing generic S-parameter shortcuts. Signal-style files should expose `RL`, `IL`, `NEXT`, and `FEXT`; power-style files should expose `mag(Zxx)`.

## Classification

The backend classifies each loaded network and returns the result in upload/load/list responses.

- `power`: any strong evidence from port names (`VDD`, `VCC`, `VSS`, `GND`, `PWR`, `POWER`, `PDN`, `VRM`) or reference impedance `z0 <= 1 ohm`.
- `signal`: any strong evidence from port names (`TX`, `RX`, `DP`, `DN`, `CH`, `LINE`, `NET`, `P`, `N`) or reference impedance `z0 >= 25 ohm`.
- `mixed`: both signal and power evidence are present.
- `unknown`: no reliable evidence.

For `mixed` and `unknown`, the UI keeps existing generic shortcuts unless a recommended action has enough data.

## Port Pair Detection

Signal actions depend on identifying two ends of the same net.

1. Name-based pairing comes first. Normalize port names by removing common endpoint/prefix tokens such as `J1`, `J2`, `NEAR`, `FAR`, `IN`, `OUT`, `TX`, `RX`, separators, and case. Ports with the same remaining net key are candidate endpoints.
2. Matrix-based pairing fills gaps. At the first frequency point, for each port `x`, choose the non-self port `y` with the largest `Sxy_dB`, meaning the value closest to `0 dB`. If `x` and `y` mutually choose each other, treat them as a high-confidence pair.
3. Non-mutual matches may be returned as lower-confidence candidates but should not drive default buttons unless no high-confidence pairs exist.

The backend returns `port_pairs` with indices, labels, source (`name` or `matrix`), and confidence.

## Quick Actions

The backend returns concrete parameter lists for each action.

- `RL`: all reflection parameters `Sii`.
- `IL`: transmission between each detected endpoint pair, using both `Sxy` and `Syx` only when both directions are useful.
- `NEXT`: near-end crosstalk. After endpoint pairs are known, group one endpoint from each pair as the near side and combine near-side ports across different nets.
- `FEXT`: far-end crosstalk. Combine one net's near-side source with another net's far-side endpoint.
- `mag(Zxx)`: for power networks, plot impedance magnitude for diagonal `Zii` terms rather than S-parameter dB.

When near/far orientation cannot be inferred from names, use pair order from matrix pairing consistently and mark the action as inferred.

## UI Behavior

`dashboard.html` renders quick buttons from backend-provided `quick_actions`. Clicking a button selects its recommended parameters and chart mode. Existing `全部`, `反射`, `传输`, and `端口N` shortcuts remain as fallback for unsupported or ambiguous files.

## Testing

Add unit tests for classification, name-based pairing, matrix-based pairing by closest-to-zero dB, quick-action parameter generation, and Flask upload/load response metadata. Add focused frontend tests only if the project already gains a JavaScript test harness; otherwise validate rendered button HTML through small pure JS helper functions or keep the logic backend-driven.
