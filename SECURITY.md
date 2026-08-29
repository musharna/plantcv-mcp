# Security Policy

## Read scope

The server reads image files from any path the host user can read unless
`--root DIR` / `PLANTCV_MCP_ROOTS` confines it; see README "Restricting what the
server may read". Report a path that escapes a configured root as a vulnerability.

## Supported versions

`plantcv-mcp` ships fixes against the latest released version only. The current
release is **v1.5.4**. Please reproduce any issue on the latest release
(`uvx plantcv-mcp` always pulls it) before reporting.

| Version         | Supported          |
| --------------- | ------------------ |
| latest (1.0.x) | :white_check_mark: |
| < latest        | :x:                |

## Reporting a vulnerability

**Please do not open a public issue for a security vulnerability.**

Report privately, either way:

- Preferred: use GitHub's **"Report a vulnerability"** button under the repo's
  **Security** tab (private security advisories), or
- Email **mjarnold1998@gmail.com**.

Please include a description of the issue, the affected version, and a minimal
reproduction. You can expect an initial acknowledgement within a few days. Once a
fix ships, you'll be credited in the release notes unless you ask otherwise.

## Security model

**This server reads image files anywhere on the host filesystem and returns them
to the model as images.** That is the trust boundary, and it is deliberate rather
than accidental — the README's "Security and trust boundary" section documents it
in full and is the authoritative description. In brief:

- `suggest_segmentation`, `segment`, `measure` and the batch tools take an
  `image_path` and pass it to PlantCV's reader. **There is no directory allow-list
  and no sandbox.** Any path the operating-system user running the server can read
  may be decoded and returned as a base64 image in the model's context.
- A prompt-injected or adversarial model can therefore use it to view arbitrary
  **image** files on the machine. Non-image files fail to decode and raise, but the
  error discloses whether the path exists.
- **Do not run it as root**, and do not expose it to untrusted input on a machine
  holding sensitive imagery. Run it as a user whose read access you are comfortable
  exposing to the model driving it.
- **No network access.** The server performs no outbound requests.

Restricting reads to a configured root directory is a candidate for a future
release. It is deliberately **not** implemented today; this section exists so that
is a decision you make rather than a surprise you discover.

Reports that the server reads a path the caller asked it to read are **working as
documented**, not vulnerabilities. Reports that it reads a path the caller did
**not** ask for, discloses non-image file contents, or escapes the running user's
own permissions are in scope and welcome.
