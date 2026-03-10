"""Single source of truth for the mlplatform package version.

Release process
---------------
Use bump-my-version (from mlplatform[dev])::

    cd mlplatform
    bump-my-version bump patch   # or minor, major

This updates this file, commits, and creates the tag. Then::

    git push origin main && git push origin vX.Y.Z

GitHub Actions picks up the tag and automatically builds and publishes the
package.  The workflow validates that the tag matches the version declared here.
"""

__version__ = "0.1.0"
