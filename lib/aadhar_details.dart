import 'package:flutter/material.dart';
import 'user_details.dart';
import 'services/backend_service.dart';

class AadharSearchPage extends StatefulWidget {
  @override
  _AadharSearchPageState createState() => _AadharSearchPageState();
}

class _AadharSearchPageState extends State<AadharSearchPage> {
  final BackendService _backendService = BackendService();
  List<Map<String, dynamic>> _allUsers = [];
  List<Map<String, dynamic>> _filteredUsers = [];
  final TextEditingController _searchController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _fetchAllUsers();
    _searchController.addListener(_filterAadharNumbers);
  }

  Future<void> _fetchAllUsers() async {
    try {
      List<Map<String, dynamic>> users = await _backendService.getAllUsers();
      setState(() {
        _allUsers = users;
        _filteredUsers = users;
      });
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error fetching users: $e')),
      );
    }
  }

  void _filterAadharNumbers() {
    final query = _searchController.text.toLowerCase();
    setState(() {
      _filteredUsers = _allUsers
          .where((user) {
            // Safely handle null original_aadhaar_number
            final aadhar = user['original_aadhaar_number'] as String? ?? '';
            return aadhar.toLowerCase().contains(query);
          })
          .toList();
    });
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.blueGrey.shade200, // or any color you find appealing
        title: Text('Aadhar Search'),
        leading: IconButton(
          icon: Icon(Icons.arrow_back),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.purple.shade700, Colors.blue.shade900],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.only(top: 20, right: 80),
              child: Align(
                alignment: Alignment.topRight,
                child: Container(
                  width: 250,
                  child: TextField(
                    controller: _searchController,
                    decoration: InputDecoration(
                      hintText: "Search Aadhar Number",
                      filled: true,
                      fillColor: Colors.white.withOpacity(0.2),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(8),
                        borderSide: BorderSide.none,
                      ),
                      prefixIcon: Icon(
                        Icons.search,
                        color: Colors.white,
                      ),
                    ),
                    style: TextStyle(color: Colors.white),
                  ),
                ),
              ),
            ),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.all(20.0),
                child: ListView.builder(
                  itemCount: _filteredUsers.length,
                  itemBuilder: (context, index) {
                    return GestureDetector(
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => UserDetailsPage(
                              user: _filteredUsers[index],
                            ),
                          ),
                        );
                      },
                      child: Card(
                        color: Colors.white.withOpacity(0.1),
                        margin: EdgeInsets.symmetric(vertical: 8),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Padding(
                          padding: const EdgeInsets.all(12.0),
                          child: Text(
                            _filteredUsers[index]['original_aadhaar_number'] as String? ?? 'N/A',
                            style: TextStyle(
                              fontSize: 18,
                              color: Colors.white,
                            ),
                          ),
                        ),
                      ),
                    );
                  },
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}