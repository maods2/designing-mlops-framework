"""Single source of truth for the mlplatform package version.

Release process
---------------
1. Update ``__version__`` below.
2. Commit: ``git commit -m "chore: bump version to X.Y.Z"``
3. Push a tag: ``git tag vX.Y.Z && git push origin vX.Y.Z``

GitHub Actions picks up the tag and automatically builds and publishes the
package.  The workflow also validates that the tag matches the version declared
here, so the tag and the code are always in sync.
"""

__version__ = "0.1.0"
