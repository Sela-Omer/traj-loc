## Contributing

Bug reports and pull requests are welcome.

**Reporting a bug.** Open an issue with the command you ran, its full output, and your GPU, CUDA and
driver versions. For a problem with a scene, attach the `scene.json` and its image.

**Pull requests.** Keep each one focused on one change. A change that can alter generated videos or
trained weights (the attention supervision, the heatmaps, the preprocessing, the samplers, a dependency
in `requirements/`) should say so, and show that `trajloc generate` on the examples gives the same videos
before and after it.

**Security.** Report security issues as described in [SECURITY.md](SECURITY.md), not in a public issue.

**Code of conduct.** This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
