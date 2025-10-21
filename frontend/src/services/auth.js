import axios from 'axios';

// --- ✅ Set your deployed backend URL here ---
const RAW_BASE = 'https://maya-16d-3.onrender.com'; // Your Render backend
const API_BASE = `${RAW_BASE}/api`;
const API_AUTH = `${API_BASE}/auth/`;

// Legacy fallback (older backends expose /auth/* without /api prefix)
const LEGACY_AUTH = `${RAW_BASE}/auth/`;

const authService = {
  // --- ✅ Register ---
  register(email, password) {
    return axios
      .post(API_AUTH + 'register', { email, password })
      .catch(async (err) => {
        try {
          return await axios.post(LEGACY_AUTH + 'register', { email, password });
        } catch (e) {
          throw err;
        }
      });
  },

  // --- ✅ Login ---
  login(email, password) {
    const payload = { email, password };

    const doPost = (base) =>
      axios.post(base + 'login', payload, {
        headers: {
          'Content-Type': 'application/json',
        },
      });

    return doPost(API_AUTH).catch(async (err) => {
      try {
        return await doPost(LEGACY_AUTH);
      } catch (e) {
        throw err;
      }
    });
  },

  // --- ✅ Update Profile (local only for now) ---
  updateProfile(data) {
    const user = JSON.parse(localStorage.getItem('user')) || {};
    const updatedUser = { ...user, ...data };
    localStorage.setItem('user', JSON.stringify(updatedUser));
    return Promise.resolve({ data: updatedUser });
  },

  // --- ✅ Token Helpers ---
  storeTokens(tokens) {
    localStorage.setItem('user', JSON.stringify(tokens));
  },

  getCurrentUser() {
    return JSON.parse(localStorage.getItem('user'));
  },

  logout() {
    localStorage.removeItem('user');
  },
};

export default authService;
