(function () {
  function cellText(row, index) {
    var cell = row.cells[index];
    return cell ? cell.textContent.trim() : "";
  }

  function parseNumber(text) {
    if (!text || text === "—" || text === "-" || text === "–") return null;
    var match = text.replace(/,/g, "").match(/[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?/);
    return match ? Number(match[0]) : null;
  }

  function compare(a, b, type, dir) {
    var emptyA = a === "" || a === "—" || a === "-" || a === "–";
    var emptyB = b === "" || b === "—" || b === "-" || b === "–";
    if (emptyA && emptyB) return 0;
    if (emptyA) return 1;
    if (emptyB) return -1;

    var cmp;
    if (type === "number") {
      var na = parseNumber(a);
      var nb = parseNumber(b);
      if (na === null && nb === null) cmp = a.localeCompare(b, undefined, { sensitivity: "base" });
      else if (na === null) return 1;
      else if (nb === null) return -1;
      else cmp = na - nb;
    } else {
      cmp = a.localeCompare(b, undefined, { sensitivity: "base", numeric: true });
    }
    return dir === "asc" ? cmp : -cmp;
  }

  function sortTable(table, colIndex, type, dir) {
    var tbody = table.tBodies[0];
    if (!tbody) return;
    var rows = Array.prototype.slice.call(tbody.rows);
    rows.sort(function (ra, rb) {
      return compare(cellText(ra, colIndex), cellText(rb, colIndex), type, dir);
    });
    rows.forEach(function (row) {
      tbody.appendChild(row);
    });
  }

  function clearSortState(table, except) {
    Array.prototype.forEach.call(table.querySelectorAll("th.sortable-col"), function (th) {
      if (th !== except) {
        th.removeAttribute("aria-sort");
        th.classList.remove("sort-asc", "sort-desc");
      }
    });
  }

  function enhanceTable(table) {
    var headers = table.querySelectorAll("thead th");
    Array.prototype.forEach.call(headers, function (th, index) {
      th.classList.add("sortable-col");
      th.setAttribute("tabindex", "0");
      th.setAttribute("role", "columnheader");
      th.setAttribute("aria-sort", "none");
      if (!th.querySelector(".sort-label")) {
        var label = document.createElement("span");
        label.className = "sort-label";
        while (th.firstChild) label.appendChild(th.firstChild);
        th.appendChild(label);
        var indicator = document.createElement("span");
        indicator.className = "sort-indicator";
        indicator.setAttribute("aria-hidden", "true");
        th.appendChild(indicator);
      }

      function activate() {
        var type = th.getAttribute("data-type") || (th.classList.contains("num") ? "number" : "text");
        var current = th.getAttribute("aria-sort");
        var dir = current === "ascending" ? "desc" : "asc";
        clearSortState(table, th);
        th.setAttribute("aria-sort", dir === "asc" ? "ascending" : "descending");
        th.classList.toggle("sort-asc", dir === "asc");
        th.classList.toggle("sort-desc", dir === "desc");
        sortTable(table, index, type, dir);
      }

      th.addEventListener("click", activate);
      th.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          activate();
        }
      });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("table.sortable").forEach(enhanceTable);
  });
})();
