#!/usr/bin/env bash
# =============================================================================
# push_seguro.sh "<mensaje de commit>" <ruta propia> [<ruta propia> ...]
#
# Commitea SOLO las rutas que este job posee y las sube a la rama actual sin pisar
# el trabajo de nadie:
#   - rebase sobre origin/<rama> (nunca `pull -X ours`, que descartaba commits ajenos
#     en archivos binarios como la base de datos);
#   - si un archivo propio entra en conflicto, se queda la versión de este job;
#   - si entra en conflicto cualquier otro archivo, se queda la versión de origin;
#   - hasta 3 reintentos si alguien empuja mientras tanto.
# =============================================================================
set -euo pipefail

MENSAJE="${1:?Falta el mensaje de commit}"
shift
if [ "$#" -eq 0 ]; then
  echo "❌ Indica al menos una ruta propia" >&2
  exit 2
fi
RUTAS_PROPIAS=("$@")
RAMA="$(git rev-parse --abbrev-ref HEAD)"

git config --local user.email "action@github.com"
git config --local user.name "GitHub Action"

git add -- "${RUTAS_PROPIAS[@]}"
if git diff --cached --quiet; then
  echo "No hay cambios para commit en: ${RUTAS_PROPIAS[*]}"
  exit 0
fi
git commit -m "$MENSAJE"

es_propia() {
  local archivo="$1" ruta
  for ruta in "${RUTAS_PROPIAS[@]}"; do
    ruta="${ruta%/}"
    if [ "$archivo" = "$ruta" ] || [[ "$archivo" == "$ruta"/* ]]; then
      return 0
    fi
  done
  return 1
}

resolver_conflictos() {
  # Durante un rebase, "theirs" es el commit que se está reaplicando (el nuestro)
  # y "ours" es la rama sobre la que se rebasa (origin).
  local archivo
  while IFS= read -r archivo; do
    [ -z "$archivo" ] && continue
    if es_propia "$archivo"; then
      echo "   ↪ conflicto en $archivo: se conserva la versión de este job"
      git checkout --theirs -- "$archivo"
    else
      echo "   ↪ conflicto en $archivo: se conserva la versión de origin"
      git checkout --ours -- "$archivo"
    fi
    git add -- "$archivo"
  done < <(git diff --name-only --diff-filter=U)
}

for intento in 1 2 3; do
  git fetch origin "$RAMA"
  if ! git rebase "origin/$RAMA"; then
    while [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ]; do
      resolver_conflictos
      if ! GIT_EDITOR=true git rebase --continue; then
        # Si el commit quedó vacío tras resolver, se omite.
        if git diff --cached --quiet; then
          git rebase --skip || true
        fi
      fi
    done
  fi
  if git push origin "HEAD:$RAMA"; then
    echo "✅ Push exitoso en intento $intento"
    exit 0
  fi
  echo "⚠️ Push rechazado, reintentando ($intento/3)..."
  sleep 5
done

echo "❌ No se pudo hacer push tras 3 intentos" >&2
exit 1
