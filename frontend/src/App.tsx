import { Navigate, NavLink, Outlet, Route, Routes } from "react-router-dom";
import { useAuth } from "./auth/AuthContext";
import { ProtectedRoute } from "./auth/ProtectedRoute";
import { AdminDocumentsPage } from "./pages/AdminDocumentsPage";
import { AdminUsersPage } from "./pages/AdminUsersPage";
import { LoginPage, RegisterPage } from "./pages/AuthPages";
import { ConsultationPage } from "./pages/ConsultationPage";
import { ConsultationsPage } from "./pages/ConsultationsPage";
import { ProfilePage } from "./pages/ProfilePage";

function Layout() {
  const { user, logout } = useAuth();
  return (
    <div className="app">
      <header className="topbar">
        <NavLink to="/" className="brand">
          <img src="/favicon.svg" alt="" className="brand-icon" /> Get Best Bike
        </NavLink>
        <nav>
          <NavLink to="/" end>
            Beratungen
          </NavLink>
          {user?.role === "admin" && (
            <>
              <NavLink to="/admin/users">Nutzer</NavLink>
              <NavLink to="/admin/documents">Wissensbasis</NavLink>
            </>
          )}
          <NavLink to="/profile">{user?.display_name}</NavLink>
          <button type="button" className="link" onClick={logout}>
            Abmelden
          </button>
        </nav>
      </header>
      <main>
        <Outlet />
      </main>
    </div>
  );
}

export function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />
      <Route element={<ProtectedRoute />}>
        <Route element={<Layout />}>
          <Route index element={<ConsultationsPage />} />
          <Route path="consultations/:id" element={<ConsultationPage />} />
          <Route path="profile" element={<ProfilePage />} />
          <Route element={<ProtectedRoute adminOnly />}>
            <Route path="admin/users" element={<AdminUsersPage />} />
            <Route path="admin/documents" element={<AdminDocumentsPage />} />
          </Route>
        </Route>
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
