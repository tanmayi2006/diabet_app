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
        SnackBar(
          content: Text(
            'Error fetching users: $e',
            style: TextStyle(color: Colors.black),
          ),
          backgroundColor: Colors.white,
        ),
      );
    }
  }

  void _filterAadharNumbers() {
    final query = _searchController.text.toLowerCase();
    setState(() {
      _filteredUsers = _allUsers
          .where((user) {
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
        backgroundColor: Colors.deepPurpleAccent,
        title: Text(
          'Aadhar Search',
          style: TextStyle(color: Colors.black),
        ),
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: Colors.black),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: Container(
        color: Colors.white,
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 20),
              child: SizedBox(
                width: double.infinity, // Expands the search bar fully
                child: TextField(
                  controller: _searchController,
                  decoration: InputDecoration(
                    hintText: "Search Aadhar Number",
                    hintStyle: TextStyle(color: Colors.black54),
                    filled: true,
                    fillColor: Colors.white,
                    contentPadding: EdgeInsets.symmetric(vertical: 12, horizontal: 16),
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                      borderSide: BorderSide(color: Colors.black),
                    ),
                    prefixIcon: Icon(
                      Icons.search,
                      color: Colors.black,
                    ),
                  ),
                  style: TextStyle(color: Colors.black),
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
                        color: Colors.white,
                        margin: EdgeInsets.symmetric(vertical: 8),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                          side: BorderSide(color: Colors.black),
                        ),
                        child: Padding(
                          padding: const EdgeInsets.all(10.0),
                              child: Padding(
                          padding: const EdgeInsets.all(8.0),
                          child: Text(
                            _filteredUsers[index]['original_aadhaar_number'] as String? ?? 'N/A',
                            style: TextStyle(
                              fontSize: 18,
                              color: Colors.black, // Text inside card in black
                              fontWeight: FontWeight.bold,
                            
                              ),
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
